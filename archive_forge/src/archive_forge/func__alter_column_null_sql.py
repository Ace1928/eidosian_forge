from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.models import NOT_PROVIDED, F, UniqueConstraint
from django.db.models.constants import LOOKUP_SEP
def _alter_column_null_sql(self, model, old_field, new_field):
    if new_field.db_default is NOT_PROVIDED:
        return super()._alter_column_null_sql(model, old_field, new_field)
    new_db_params = new_field.db_parameters(connection=self.connection)
    type_sql = self._set_field_new_type(new_field, new_db_params['type'])
    return ('MODIFY %(column)s %(type)s' % {'column': self.quote_name(new_field.column), 'type': type_sql}, [])