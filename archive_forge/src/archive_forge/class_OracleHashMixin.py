from django.db import NotSupportedError
from django.db.models.expressions import Func, Value
from django.db.models.fields import CharField, IntegerField, TextField
from django.db.models.functions import Cast, Coalesce
from django.db.models.lookups import Transform
class OracleHashMixin:

    def as_oracle(self, compiler, connection, **extra_context):
        return super().as_sql(compiler, connection, template="LOWER(RAWTOHEX(STANDARD_HASH(UTL_I18N.STRING_TO_RAW(%(expressions)s, 'AL32UTF8'), '%(function)s')))", **extra_context)