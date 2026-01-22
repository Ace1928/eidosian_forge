from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.models import NOT_PROVIDED, F, UniqueConstraint
from django.db.models.constants import LOOKUP_SEP
def _comment_sql(self, comment):
    comment_sql = super()._comment_sql(comment)
    return f' COMMENT {comment_sql}'