from pygments.lexer import RegexLexer, words
from pygments.token import Text, Comment, Keyword, Name, Operator, Whitespace
class SrcinfoLexer(RegexLexer):
    """Lexer for .SRCINFO files used by Arch Linux Packages.

    .. versionadded:: 2.11
    """
    name = 'Srcinfo'
    aliases = ['srcinfo']
    filenames = ['.SRCINFO']
    tokens = {'root': [('\\s+', Whitespace), ('#.*', Comment.Single), (words(keywords), Keyword, 'assignment'), (words(architecture_dependent_keywords, suffix='_\\w+'), Keyword, 'assignment'), ('\\w+', Name.Variable, 'assignment')], 'assignment': [(' +', Whitespace), ('=', Operator, 'value')], 'value': [(' +', Whitespace), ('.*', Text, '#pop:2')]}