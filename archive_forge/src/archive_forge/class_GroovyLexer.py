import re
from pygments.lexer import Lexer, RegexLexer, include, bygroups, using, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import shebang_matches
from pygments import unistring as uni
class GroovyLexer(RegexLexer):
    """
    For `Groovy <http://groovy.codehaus.org/>`_ source code.

    .. versionadded:: 1.5
    """
    name = 'Groovy'
    aliases = ['groovy']
    filenames = ['*.groovy', '*.gradle']
    mimetypes = ['text/x-groovy']
    flags = re.MULTILINE | re.DOTALL
    tokens = {'root': [('#!(.*?)$', Comment.Preproc, 'base'), default('base')], 'base': [('^(\\s*(?:[a-zA-Z_][\\w.\\[\\]]*\\s+)+?)([a-zA-Z_]\\w*)(\\s*)(\\()', bygroups(using(this), Name.Function, Text, Operator)), ('[^\\S\\n]+', Text), ('//.*?\\n', Comment.Single), ('/\\*.*?\\*/', Comment.Multiline), ('@[a-zA-Z_][\\w.]*', Name.Decorator), ('(assert|break|case|catch|continue|default|do|else|finally|for|if|goto|instanceof|new|return|switch|this|throw|try|while|in|as)\\b', Keyword), ('(abstract|const|enum|extends|final|implements|native|private|protected|public|static|strictfp|super|synchronized|throws|transient|volatile)\\b', Keyword.Declaration), ('(def|boolean|byte|char|double|float|int|long|short|void)\\b', Keyword.Type), ('(package)(\\s+)', bygroups(Keyword.Namespace, Text)), ('(true|false|null)\\b', Keyword.Constant), ('(class|interface)(\\s+)', bygroups(Keyword.Declaration, Text), 'class'), ('(import)(\\s+)', bygroups(Keyword.Namespace, Text), 'import'), ('""".*?"""', String.Double), ("'''.*?'''", String.Single), ('"(\\\\\\\\|\\\\"|[^"])*"', String.Double), ("'(\\\\\\\\|\\\\'|[^'])*'", String.Single), ('\\$/((?!/\\$).)*/\\$', String), ('/(\\\\\\\\|\\\\"|[^/])*/', String), ("'\\\\.'|'[^\\\\]'|'\\\\u[0-9a-fA-F]{4}'", String.Char), ('(\\.)([a-zA-Z_]\\w*)', bygroups(Operator, Name.Attribute)), ('[a-zA-Z_]\\w*:', Name.Label), ('[a-zA-Z_$]\\w*', Name), ('[~^*!%&\\[\\](){}<>|+=:;,./?-]', Operator), ('[0-9][0-9]*\\.[0-9]+([eE][0-9]+)?[fd]?', Number.Float), ('0x[0-9a-fA-F]+', Number.Hex), ('[0-9]+L?', Number.Integer), ('\\n', Text)], 'class': [('[a-zA-Z_]\\w*', Name.Class, '#pop')], 'import': [('[\\w.]+\\*?', Name.Namespace, '#pop')]}

    def analyse_text(text):
        return shebang_matches(text, 'groovy')