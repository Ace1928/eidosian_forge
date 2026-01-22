from pygments.lexer import RegexLexer, bygroups
from pygments.token import Comment, Keyword, Operator, Name, String, Number, \
class GraphvizLexer(RegexLexer):
    """
    For graphviz DOT graph description language.

    .. versionadded:: 2.8
    """
    name = 'Graphviz'
    url = 'https://www.graphviz.org/doc/info/lang.html'
    aliases = ['graphviz', 'dot']
    filenames = ['*.gv', '*.dot']
    mimetypes = ['text/x-graphviz', 'text/vnd.graphviz']
    tokens = {'root': [('\\s+', Whitespace), ('(#|//).*?$', Comment.Single), ('/(\\\\\\n)?[*](.|\\n)*?[*](\\\\\\n)?/', Comment.Multiline), ('(?i)(node|edge|graph|digraph|subgraph|strict)\\b', Keyword), ('--|->', Operator), ('[{}[\\]:;,]', Punctuation), ('(\\b\\D\\w*)(\\s*)(=)(\\s*)', bygroups(Name.Attribute, Whitespace, Punctuation, Whitespace), 'attr_id'), ('\\b(n|ne|e|se|s|sw|w|nw|c|_)\\b', Name.Builtin), ('\\b\\D\\w*', Name.Tag), ('[-]?((\\.[0-9]+)|([0-9]+(\\.[0-9]*)?))', Number), ('"(\\\\"|[^"])*?"', Name.Tag), ('<', Punctuation, 'xml')], 'attr_id': [('\\b\\D\\w*', String, '#pop'), ('[-]?((\\.[0-9]+)|([0-9]+(\\.[0-9]*)?))', Number, '#pop'), ('"(\\\\"|[^"])*?"', String.Double, '#pop'), ('<', Punctuation, ('#pop', 'xml'))], 'xml': [('<', Punctuation, '#push'), ('>', Punctuation, '#pop'), ('\\s+', Whitespace), ('[^<>\\s]', Name.Tag)]}