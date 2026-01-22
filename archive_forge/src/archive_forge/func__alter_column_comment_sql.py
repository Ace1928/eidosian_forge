from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.models import NOT_PROVIDED, F, UniqueConstraint
from django.db.models.constants import LOOKUP_SEP
def _alter_column_comment_sql(self, model, new_field, new_type, new_db_comment):
    return ('', [])