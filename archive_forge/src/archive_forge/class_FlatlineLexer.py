import re
from pygments.lexer import ExtendedRegexLexer, RegexLexer, bygroups, words, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class FlatlineLexer(RegexLexer):
    """
    Lexer for `Flatline <https://github.com/bigmlcom/flatline>`_ expressions.

    .. versionadded:: 2.2
    """
    name = 'Flatline'
    aliases = ['flatline']
    filenames = []
    mimetypes = ['text/x-flatline']
    special_forms = ('let',)
    builtins = ('!=', '*', '+', '-', '<', '<=', '=', '>', '>=', 'abs', 'acos', 'all', 'all-but', 'all-with-defaults', 'all-with-numeric-default', 'and', 'asin', 'atan', 'avg', 'avg-window', 'bin-center', 'bin-count', 'call', 'category-count', 'ceil', 'cond', 'cond-window', 'cons', 'cos', 'cosh', 'count', 'diff-window', 'div', 'ensure-value', 'ensure-weighted-value', 'epoch', 'epoch-day', 'epoch-fields', 'epoch-hour', 'epoch-millisecond', 'epoch-minute', 'epoch-month', 'epoch-second', 'epoch-weekday', 'epoch-year', 'exp', 'f', 'field', 'field-prop', 'fields', 'filter', 'first', 'floor', 'head', 'if', 'in', 'integer', 'language', 'length', 'levenshtein', 'linear-regression', 'list', 'ln', 'log', 'log10', 'map', 'matches', 'matches?', 'max', 'maximum', 'md5', 'mean', 'median', 'min', 'minimum', 'missing', 'missing-count', 'missing?', 'missing_count', 'mod', 'mode', 'normalize', 'not', 'nth', 'occurrences', 'or', 'percentile', 'percentile-label', 'population', 'population-fraction', 'pow', 'preferred', 'preferred?', 'quantile-label', 'rand', 'rand-int', 'random-value', 're-quote', 'real', 'replace', 'replace-first', 'rest', 'round', 'row-number', 'segment-label', 'sha1', 'sha256', 'sin', 'sinh', 'sqrt', 'square', 'standard-deviation', 'standard_deviation', 'str', 'subs', 'sum', 'sum-squares', 'sum-window', 'sum_squares', 'summary', 'summary-no', 'summary-str', 'tail', 'tan', 'tanh', 'to-degrees', 'to-radians', 'variance', 'vectorize', 'weighted-random-value', 'window', 'winnow', 'within-percentiles?', 'z-score')
    valid_name = '(?!#)[\\w!$%*+<=>?/.#-]+'
    tokens = {'root': [('[,\\s]+', Text), ('-?\\d+\\.\\d+', Number.Float), ('-?\\d+', Number.Integer), ('0x-?[a-f\\d]+', Number.Hex), ('"(\\\\\\\\|\\\\"|[^"])*"', String), ('\\\\(.|[a-z]+)', String.Char), ('_', String.Symbol), (words(special_forms, suffix=' '), Keyword), (words(builtins, suffix=' '), Name.Builtin), ('(?<=\\()' + valid_name, Name.Function), (valid_name, Name.Variable), ('(\\(|\\))', Punctuation)]}