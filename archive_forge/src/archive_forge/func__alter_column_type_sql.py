from django.db.backends.postgresql.schema import DatabaseSchemaEditor
from django.db.models.expressions import Col, Func
def _alter_column_type_sql(self, table, old_field, new_field, new_type, old_collation, new_collation):
    """
        Special case when dimension changed.
        """
    if not hasattr(old_field, 'dim') or not hasattr(new_field, 'dim'):
        return super()._alter_column_type_sql(table, old_field, new_field, new_type, old_collation, new_collation)
    if old_field.dim == 2 and new_field.dim == 3:
        sql_alter = self.sql_alter_column_to_3d
    elif old_field.dim == 3 and new_field.dim == 2:
        sql_alter = self.sql_alter_column_to_2d
    else:
        sql_alter = self.sql_alter_column_type
    return ((sql_alter % {'column': self.quote_name(new_field.column), 'type': new_type, 'collation': ''}, []), [])