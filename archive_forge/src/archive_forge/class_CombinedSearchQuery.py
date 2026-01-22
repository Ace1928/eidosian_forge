from django.db.models import (
from django.db.models.expressions import CombinedExpression, register_combinable_fields
from django.db.models.functions import Cast, Coalesce
class CombinedSearchQuery(SearchQueryCombinable, CombinedExpression):

    def __init__(self, lhs, connector, rhs, config, output_field=None):
        self.config = config
        super().__init__(lhs, connector, rhs, output_field)

    def __str__(self):
        return '(%s)' % super().__str__()