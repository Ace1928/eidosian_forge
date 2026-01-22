import copy
from decimal import Decimal
from django.apps.registry import Apps
from django.db import NotSupportedError
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.backends.ddl_references import Statement
from django.db.backends.utils import strip_quotes
from django.db.models import NOT_PROVIDED, UniqueConstraint
def _remake_table(self, model, create_field=None, delete_field=None, alter_fields=None):
    """
        Shortcut to transform a model from old_model into new_model

        This follows the correct procedure to perform non-rename or column
        addition operations based on SQLite's documentation

        https://www.sqlite.org/lang_altertable.html#caution

        The essential steps are:
          1. Create a table with the updated definition called "new__app_model"
          2. Copy the data from the existing "app_model" table to the new table
          3. Drop the "app_model" table
          4. Rename the "new__app_model" table to "app_model"
          5. Restore any index of the previous "app_model" table.
        """

    def is_self_referential(f):
        return f.is_relation and f.remote_field.model is model
    body = {f.name: f.clone() if is_self_referential(f) else f for f in model._meta.local_concrete_fields}
    mapping = {f.column: self.quote_name(f.column) for f in model._meta.local_concrete_fields if f.generated is False}
    rename_mapping = {}
    restore_pk_field = None
    alter_fields = alter_fields or []
    if getattr(create_field, 'primary_key', False) or any((getattr(new_field, 'primary_key', False) for _, new_field in alter_fields)):
        for name, field in list(body.items()):
            if field.primary_key and (not any((name == new_field.name for _, new_field in alter_fields))):
                field.primary_key = False
                restore_pk_field = field
                if field.auto_created:
                    del body[name]
                    del mapping[field.column]
    if create_field:
        body[create_field.name] = create_field
        if create_field.db_default is NOT_PROVIDED and (not (create_field.many_to_many or create_field.generated)) and create_field.concrete:
            mapping[create_field.column] = self.prepare_default(self.effective_default(create_field))
    for alter_field in alter_fields:
        old_field, new_field = alter_field
        body.pop(old_field.name, None)
        mapping.pop(old_field.column, None)
        body[new_field.name] = new_field
        if old_field.null and (not new_field.null):
            if new_field.db_default is NOT_PROVIDED:
                default = self.prepare_default(self.effective_default(new_field))
            else:
                default, _ = self.db_default_sql(new_field)
            case_sql = 'coalesce(%(col)s, %(default)s)' % {'col': self.quote_name(old_field.column), 'default': default}
            mapping[new_field.column] = case_sql
        else:
            mapping[new_field.column] = self.quote_name(old_field.column)
        rename_mapping[old_field.name] = new_field.name
    if delete_field:
        del body[delete_field.name]
        mapping.pop(delete_field.column, None)
        if delete_field.many_to_many and delete_field.remote_field.through._meta.auto_created:
            return self.delete_model(delete_field.remote_field.through)
    apps = Apps()
    unique_together = [[rename_mapping.get(n, n) for n in unique] for unique in model._meta.unique_together]
    index_together = [[rename_mapping.get(n, n) for n in index] for index in model._meta.index_together]
    indexes = model._meta.indexes
    if delete_field:
        indexes = [index for index in indexes if delete_field.name not in index.fields]
    constraints = list(model._meta.constraints)
    body_copy = copy.deepcopy(body)
    meta_contents = {'app_label': model._meta.app_label, 'db_table': model._meta.db_table, 'unique_together': unique_together, 'index_together': index_together, 'indexes': indexes, 'constraints': constraints, 'apps': apps}
    meta = type('Meta', (), meta_contents)
    body_copy['Meta'] = meta
    body_copy['__module__'] = model.__module__
    type(model._meta.object_name, model.__bases__, body_copy)
    body_copy = copy.deepcopy(body)
    meta_contents = {'app_label': model._meta.app_label, 'db_table': 'new__%s' % strip_quotes(model._meta.db_table), 'unique_together': unique_together, 'index_together': index_together, 'indexes': indexes, 'constraints': constraints, 'apps': apps}
    meta = type('Meta', (), meta_contents)
    body_copy['Meta'] = meta
    body_copy['__module__'] = model.__module__
    new_model = type('New%s' % model._meta.object_name, model.__bases__, body_copy)
    self.create_model(new_model)
    self.execute('INSERT INTO %s (%s) SELECT %s FROM %s' % (self.quote_name(new_model._meta.db_table), ', '.join((self.quote_name(x) for x in mapping)), ', '.join(mapping.values()), self.quote_name(model._meta.db_table)))
    self.delete_model(model, handle_autom2m=False)
    self.alter_db_table(new_model, new_model._meta.db_table, model._meta.db_table)
    for sql in self.deferred_sql:
        self.execute(sql)
    self.deferred_sql = []
    if restore_pk_field:
        restore_pk_field.primary_key = True