from django.db.models import (
from django.db.models.expressions import CombinedExpression, register_combinable_fields
from django.db.models.functions import Cast, Coalesce
class TrigramDistance(TrigramBase):
    function = ''
    arg_joiner = ' <-> '