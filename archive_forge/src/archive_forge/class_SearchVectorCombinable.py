from django.db.models import (
from django.db.models.expressions import CombinedExpression, register_combinable_fields
from django.db.models.functions import Cast, Coalesce
class SearchVectorCombinable:
    ADD = '||'

    def _combine(self, other, connector, reversed):
        if not isinstance(other, SearchVectorCombinable):
            raise TypeError('SearchVector can only be combined with other SearchVector instances, got %s.' % type(other).__name__)
        if reversed:
            return CombinedSearchVector(other, connector, self, self.config)
        return CombinedSearchVector(self, connector, other, self.config)