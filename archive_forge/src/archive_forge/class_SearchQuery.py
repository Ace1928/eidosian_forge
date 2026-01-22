from django.db.models import (
from django.db.models.expressions import CombinedExpression, register_combinable_fields
from django.db.models.functions import Cast, Coalesce
class SearchQuery(SearchQueryCombinable, Func):
    output_field = SearchQueryField()
    SEARCH_TYPES = {'plain': 'plainto_tsquery', 'phrase': 'phraseto_tsquery', 'raw': 'to_tsquery', 'websearch': 'websearch_to_tsquery'}

    def __init__(self, value, output_field=None, *, config=None, invert=False, search_type='plain'):
        self.function = self.SEARCH_TYPES.get(search_type)
        if self.function is None:
            raise ValueError("Unknown search_type argument '%s'." % search_type)
        if not hasattr(value, 'resolve_expression'):
            value = Value(value)
        expressions = (value,)
        self.config = SearchConfig.from_parameter(config)
        if self.config is not None:
            expressions = (self.config,) + expressions
        self.invert = invert
        super().__init__(*expressions, output_field=output_field)

    def as_sql(self, compiler, connection, function=None, template=None):
        sql, params = super().as_sql(compiler, connection, function, template)
        if self.invert:
            sql = '!!(%s)' % sql
        return (sql, params)

    def __invert__(self):
        clone = self.copy()
        clone.invert = not self.invert
        return clone

    def __str__(self):
        result = super().__str__()
        return '~%s' % result if self.invert else result