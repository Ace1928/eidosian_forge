from django.db import models
from django.db.migrations.operations.base import Operation
from django.db.migrations.state import ModelState
from django.db.migrations.utils import field_references, resolve_relation
from django.db.models.options import normalize_together
from django.utils.functional import cached_property
from .fields import AddField, AlterField, FieldOperation, RemoveField, RenameField
class RenameIndex(IndexOperation):
    """Rename an index."""

    def __init__(self, model_name, new_name, old_name=None, old_fields=None):
        if not old_name and (not old_fields):
            raise ValueError('RenameIndex requires one of old_name and old_fields arguments to be set.')
        if old_name and old_fields:
            raise ValueError('RenameIndex.old_name and old_fields are mutually exclusive.')
        self.model_name = model_name
        self.new_name = new_name
        self.old_name = old_name
        self.old_fields = old_fields

    @cached_property
    def old_name_lower(self):
        return self.old_name.lower()

    @cached_property
    def new_name_lower(self):
        return self.new_name.lower()

    def deconstruct(self):
        kwargs = {'model_name': self.model_name, 'new_name': self.new_name}
        if self.old_name:
            kwargs['old_name'] = self.old_name
        if self.old_fields:
            kwargs['old_fields'] = self.old_fields
        return (self.__class__.__qualname__, [], kwargs)

    def state_forwards(self, app_label, state):
        if self.old_fields:
            state.add_index(app_label, self.model_name_lower, models.Index(fields=self.old_fields, name=self.new_name))
            state.remove_model_options(app_label, self.model_name_lower, AlterIndexTogether.option_name, self.old_fields)
        else:
            state.rename_index(app_label, self.model_name_lower, self.old_name, self.new_name)

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        model = to_state.apps.get_model(app_label, self.model_name)
        if not self.allow_migrate_model(schema_editor.connection.alias, model):
            return
        if self.old_fields:
            from_model = from_state.apps.get_model(app_label, self.model_name)
            columns = [from_model._meta.get_field(field).column for field in self.old_fields]
            matching_index_name = schema_editor._constraint_names(from_model, column_names=columns, index=True)
            if len(matching_index_name) != 1:
                raise ValueError('Found wrong number (%s) of indexes for %s(%s).' % (len(matching_index_name), from_model._meta.db_table, ', '.join(columns)))
            old_index = models.Index(fields=self.old_fields, name=matching_index_name[0])
        else:
            from_model_state = from_state.models[app_label, self.model_name_lower]
            old_index = from_model_state.get_index_by_name(self.old_name)
        if old_index.name == self.new_name:
            return
        to_model_state = to_state.models[app_label, self.model_name_lower]
        new_index = to_model_state.get_index_by_name(self.new_name)
        schema_editor.rename_index(model, old_index, new_index)

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        if self.old_fields:
            return
        self.new_name_lower, self.old_name_lower = (self.old_name_lower, self.new_name_lower)
        self.new_name, self.old_name = (self.old_name, self.new_name)
        self.database_forwards(app_label, schema_editor, from_state, to_state)
        self.new_name_lower, self.old_name_lower = (self.old_name_lower, self.new_name_lower)
        self.new_name, self.old_name = (self.old_name, self.new_name)

    def describe(self):
        if self.old_name:
            return f'Rename index {self.old_name} on {self.model_name} to {self.new_name}'
        return f'Rename unnamed index for {self.old_fields} on {self.model_name} to {self.new_name}'

    @property
    def migration_name_fragment(self):
        if self.old_name:
            return 'rename_%s_%s' % (self.old_name_lower, self.new_name_lower)
        return 'rename_%s_%s_%s' % (self.model_name_lower, '_'.join(self.old_fields), self.new_name_lower)

    def reduce(self, operation, app_label):
        if isinstance(operation, RenameIndex) and self.model_name_lower == operation.model_name_lower and operation.old_name and (self.new_name_lower == operation.old_name_lower):
            return [RenameIndex(self.model_name, new_name=operation.new_name, old_name=self.old_name, old_fields=self.old_fields)]
        return super().reduce(operation, app_label)