from pygments.lexer import RegexLexer, words, re
from pygments.token import Text, Operator, Keyword, Name, Comment
class RoboconfGraphLexer(RegexLexer):
    """
    Lexer for `Roboconf <http://roboconf.net/en/roboconf.html>`_ graph files.

    .. versionadded:: 2.1
    """
    name = 'Roboconf Graph'
    aliases = ['roboconf-graph']
    filenames = ['*.graph']
    flags = re.IGNORECASE | re.MULTILINE
    tokens = {'root': [('\\s+', Text), ('=', Operator), (words(('facet', 'import'), suffix='\\s*\\b', prefix='\\b'), Keyword), (words(('installer', 'extends', 'exports', 'imports', 'facets', 'children'), suffix='\\s*:?', prefix='\\b'), Name), ('#.*\\n', Comment), ('[^#]', Text), ('.*\\n', Text)]}