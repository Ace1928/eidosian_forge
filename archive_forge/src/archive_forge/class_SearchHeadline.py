from django.db.models import (
from django.db.models.expressions import CombinedExpression, register_combinable_fields
from django.db.models.functions import Cast, Coalesce
class SearchHeadline(Func):
    function = 'ts_headline'
    template = '%(function)s(%(expressions)s%(options)s)'
    output_field = TextField()

    def __init__(self, expression, query, *, config=None, start_sel=None, stop_sel=None, max_words=None, min_words=None, short_word=None, highlight_all=None, max_fragments=None, fragment_delimiter=None):
        if not hasattr(query, 'resolve_expression'):
            query = SearchQuery(query)
        options = {'StartSel': start_sel, 'StopSel': stop_sel, 'MaxWords': max_words, 'MinWords': min_words, 'ShortWord': short_word, 'HighlightAll': highlight_all, 'MaxFragments': max_fragments, 'FragmentDelimiter': fragment_delimiter}
        self.options = {option: value for option, value in options.items() if value is not None}
        expressions = (expression, query)
        if config is not None:
            config = SearchConfig.from_parameter(config)
            expressions = (config,) + expressions
        super().__init__(*expressions)

    def as_sql(self, compiler, connection, function=None, template=None):
        options_sql = ''
        options_params = []
        if self.options:
            options_params.append(', '.join((connection.ops.compose_sql(f'{option}=%s', [value]) for option, value in self.options.items())))
            options_sql = ', %s'
        sql, params = super().as_sql(compiler, connection, function=function, template=template, options=options_sql)
        return (sql, params + options_params)