from pygments.lexer import RegexLexer, bygroups, default, words
from pygments.token import Comment, Keyword, Name, Number, Operator, \
class PromQLLexer(RegexLexer):
    """
    For PromQL queries.

    For details about the grammar see:
    https://github.com/prometheus/prometheus/tree/master/promql/parser

    .. versionadded: 2.7
    """
    name = 'PromQL'
    url = 'https://prometheus.io/docs/prometheus/latest/querying/basics/'
    aliases = ['promql']
    filenames = ['*.promql']
    base_keywords = (words(('bool', 'by', 'group_left', 'group_right', 'ignoring', 'offset', 'on', 'without'), suffix='\\b'), Keyword)
    aggregator_keywords = (words(('sum', 'min', 'max', 'avg', 'group', 'stddev', 'stdvar', 'count', 'count_values', 'bottomk', 'topk', 'quantile'), suffix='\\b'), Keyword)
    function_keywords = (words(('abs', 'absent', 'absent_over_time', 'avg_over_time', 'ceil', 'changes', 'clamp_max', 'clamp_min', 'count_over_time', 'day_of_month', 'day_of_week', 'days_in_month', 'delta', 'deriv', 'exp', 'floor', 'histogram_quantile', 'holt_winters', 'hour', 'idelta', 'increase', 'irate', 'label_join', 'label_replace', 'ln', 'log10', 'log2', 'max_over_time', 'min_over_time', 'minute', 'month', 'predict_linear', 'quantile_over_time', 'rate', 'resets', 'round', 'scalar', 'sort', 'sort_desc', 'sqrt', 'stddev_over_time', 'stdvar_over_time', 'sum_over_time', 'time', 'timestamp', 'vector', 'year'), suffix='\\b'), Keyword.Reserved)
    tokens = {'root': [('\\n', Whitespace), ('\\s+', Whitespace), (',', Punctuation), base_keywords, aggregator_keywords, function_keywords, ('[1-9][0-9]*[smhdwy]', String), ('-?[0-9]+\\.[0-9]+', Number.Float), ('-?[0-9]+', Number.Integer), ('#.*?$', Comment.Single), ('(\\+|\\-|\\*|\\/|\\%|\\^)', Operator), ('==|!=|>=|<=|<|>', Operator), ('and|or|unless', Operator.Word), ('[_a-zA-Z][a-zA-Z0-9_]+', Name.Variable), ('(["\\\'])(.*?)(["\\\'])', bygroups(Punctuation, String, Punctuation)), ('\\(', Operator, 'function'), ('\\)', Operator), ('\\{', Punctuation, 'labels'), ('\\[', Punctuation, 'range')], 'labels': [('\\}', Punctuation, '#pop'), ('\\n', Whitespace), ('\\s+', Whitespace), (',', Punctuation), ('([_a-zA-Z][a-zA-Z0-9_]*?)(\\s*?)(=~|!=|=|!~)(\\s*?)("|\\\')(.*?)("|\\\')', bygroups(Name.Label, Whitespace, Operator, Whitespace, Punctuation, String, Punctuation))], 'range': [('\\]', Punctuation, '#pop'), ('[1-9][0-9]*[smhdwy]', String)], 'function': [('\\)', Operator, '#pop'), ('\\(', Operator, '#push'), default('#pop')]}