from pygments.lexer import RegexLexer, include, bygroups, using, this, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class VCLSnippetLexer(VCLLexer):
    """
    For Varnish Configuration Language snippets.

    .. versionadded:: 2.2
    """
    name = 'VCLSnippets'
    aliases = ['vclsnippets', 'vclsnippet']
    mimetypes = ['text/x-vclsnippet']
    filenames = []

    def analyse_text(text):
        return 0
    tokens = {'snippetspre': [('\\.\\.\\.+', Comment), ('(bereq|req|req_top|resp|beresp|obj|client|server|local|remote|storage)($|\\.\\*)', Name.Variable)], 'snippetspost': [('(backend)\\b', Keyword.Reserved)], 'root': [include('snippetspre'), inherit, include('snippetspost')]}