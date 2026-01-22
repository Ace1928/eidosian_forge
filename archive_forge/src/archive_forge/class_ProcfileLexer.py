from pygments.lexer import RegexLexer, bygroups
from pygments.token import Name, Number, String, Text, Punctuation
class ProcfileLexer(RegexLexer):
    """
    Lexer for Procfile file format.

    The format is used to run processes on Heroku or is used by Foreman or
    Honcho tools.

    .. versionadded:: 2.10
    """
    name = 'Procfile'
    url = 'https://devcenter.heroku.com/articles/procfile#procfile-format'
    aliases = ['procfile']
    filenames = ['Procfile']
    tokens = {'root': [('^([a-z]+)(:)', bygroups(Name.Label, Punctuation)), ('\\s+', Text.Whitespace), ('"[^"]*"', String), ("'[^']*'", String), ('[0-9]+', Number.Integer), ('\\$[a-zA-Z_][\\w]*', Name.Variable), ('(\\w+)(=)(\\w+)', bygroups(Name.Variable, Punctuation, String)), ('([\\w\\-\\./]+)', Text)]}