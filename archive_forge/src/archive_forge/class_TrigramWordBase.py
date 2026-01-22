from django.db.models import (
from django.db.models.expressions import CombinedExpression, register_combinable_fields
from django.db.models.functions import Cast, Coalesce
class TrigramWordBase(Func):
    output_field = FloatField()

    def __init__(self, string, expression, **extra):
        if not hasattr(string, 'resolve_expression'):
            string = Value(string)
        super().__init__(string, expression, **extra)