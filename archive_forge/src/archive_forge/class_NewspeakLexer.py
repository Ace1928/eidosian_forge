from pygments.lexer import RegexLexer, include, bygroups, default
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class NewspeakLexer(RegexLexer):
    """
    For `Newspeak <http://newspeaklanguage.org/>` syntax.

    .. versionadded:: 1.1
    """
    name = 'Newspeak'
    filenames = ['*.ns2']
    aliases = ['newspeak']
    mimetypes = ['text/x-newspeak']
    tokens = {'root': [('\\b(Newsqueak2)\\b', Keyword.Declaration), ("'[^']*'", String), ('\\b(class)(\\s+)(\\w+)(\\s*)', bygroups(Keyword.Declaration, Text, Name.Class, Text)), ('\\b(mixin|self|super|private|public|protected|nil|true|false)\\b', Keyword), ('(\\w+\\:)(\\s*)([a-zA-Z_]\\w+)', bygroups(Name.Function, Text, Name.Variable)), ('(\\w+)(\\s*)(=)', bygroups(Name.Attribute, Text, Operator)), ('<\\w+>', Comment.Special), include('expressionstat'), include('whitespace')], 'expressionstat': [('(\\d+\\.\\d*|\\.\\d+|\\d+[fF])[fF]?', Number.Float), ('\\d+', Number.Integer), (':\\w+', Name.Variable), ('(\\w+)(::)', bygroups(Name.Variable, Operator)), ('\\w+:', Name.Function), ('\\w+', Name.Variable), ('\\(|\\)', Punctuation), ('\\[|\\]', Punctuation), ('\\{|\\}', Punctuation), ('(\\^|\\+|\\/|~|\\*|<|>|=|@|%|\\||&|\\?|!|,|-|:)', Operator), ('\\.|;', Punctuation), include('whitespace'), include('literals')], 'literals': [('\\$.', String), ("'[^']*'", String), ("#'[^']*'", String.Symbol), ('#\\w+:?', String.Symbol), ('#(\\+|\\/|~|\\*|<|>|=|@|%|\\||&|\\?|!|,|-)+', String.Symbol)], 'whitespace': [('\\s+', Text), ('"[^"]*"', Comment)]}