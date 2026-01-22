import math
from django.db.models.expressions import Func, Value
from django.db.models.fields import FloatField, IntegerField
from django.db.models.functions import Cast
from django.db.models.functions.mixins import (
from django.db.models.lookups import Transform
class Cot(NumericOutputFieldMixin, Transform):
    function = 'COT'
    lookup_name = 'cot'

    def as_oracle(self, compiler, connection, **extra_context):
        return super().as_sql(compiler, connection, template='(1 / TAN(%(expressions)s))', **extra_context)