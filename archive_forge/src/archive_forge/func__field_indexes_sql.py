from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.backends.ddl_references import IndexColumns
from django.db.backends.postgresql.psycopg_any import sql
from django.db.backends.utils import strip_quotes
def _field_indexes_sql(self, model, field):
    output = super()._field_indexes_sql(model, field)
    like_index_statement = self._create_like_index_sql(model, field)
    if like_index_statement is not None:
        output.append(like_index_statement)
    return output