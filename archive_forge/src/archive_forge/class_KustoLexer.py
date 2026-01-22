from pygments.lexer import RegexLexer, words
from pygments.token import (Comment, Keyword, Name, Number, Punctuation,
class KustoLexer(RegexLexer):
    """For Kusto Query Language source code.

    .. versionadded:: 2.17
    """
    name = 'Kusto'
    aliases = ['kql', 'kusto']
    filenames = ['*.kql', '*.kusto', '.csl']
    url = 'https://learn.microsoft.com/en-us/azure/data-explorer/kusto/query'
    tokens = {'root': [('\\s+', Whitespace), (words(KUSTO_KEYWORDS, suffix='\\b'), Keyword), ('//.*', Comment), (words(KUSTO_PUNCTUATION), Punctuation), ('[^\\W\\d]\\w*', Name), ('\\d+[.]\\d*|[.]\\d+', Number.Float), ('\\d+', Number.Integer), ("'", String, 'single_string'), ('"', String, 'double_string'), ("@'", String, 'single_verbatim'), ('@"', String, 'double_verbatim'), ('```', String, 'multi_string')], 'single_string': [("'", String, '#pop'), ('\\\\.', String.Escape), ("[^'\\\\]+", String)], 'double_string': [('"', String, '#pop'), ('\\\\.', String.Escape), ('[^"\\\\]+', String)], 'single_verbatim': [("'", String, '#pop'), ("[^']+", String)], 'double_verbatim': [('"', String, '#pop'), ('[^"]+', String)], 'multi_string': [('[^`]+', String), ('```', String, '#pop'), ('`', String)]}