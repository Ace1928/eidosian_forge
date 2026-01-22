import re
from pygments.lexer import ExtendedRegexLexer, RegexLexer, default, words, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.shell import BashLexer
from pygments.lexers.data import JsonLexer
class TOMLLexer(RegexLexer):
    """
    Lexer for TOML, a simple language
    for config files.

    .. versionadded:: 2.4
    """
    name = 'TOML'
    url = 'https://github.com/toml-lang/toml'
    aliases = ['toml']
    filenames = ['*.toml', 'Pipfile', 'poetry.lock']
    tokens = {'root': [('^(\\s*)(\\[.*?\\])$', bygroups(Whitespace, Keyword)), ('[ \\t]+', Whitespace), ('\\n', Whitespace), ('#.*?$', Comment.Single), ('"(\\\\\\\\|\\\\[^\\\\]|[^"\\\\])*"', String), ("\\'\\'\\'(.*)\\'\\'\\'", String), ("\\'[^\\']*\\'", String), ('(true|false)$', Keyword.Constant), ('[a-zA-Z_][\\w\\-]*', Name), ('\\d{4}-\\d{2}-\\d{2}(?:T| )\\d{2}:\\d{2}:\\d{2}(?:Z|[-+]\\d{2}:\\d{2})', Number.Integer), ('(\\d+\\.\\d*|\\d*\\.\\d+)([eE][+-]?[0-9]+)?j?', Number.Float), ('\\d+[eE][+-]?[0-9]+j?', Number.Float), ('[+-]?(?:(inf(?:inity)?)|nan)', Number.Float), ('[+-]?\\d+', Number.Integer), ('[]{}:(),;[]', Punctuation), ('\\.', Punctuation), ('=', Operator)]}