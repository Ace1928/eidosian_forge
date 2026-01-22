import re
from pygments.lexer import RegexLexer, DelegatingLexer, bygroups, include, \
from pygments.token import Punctuation, \
from pygments.util import get_choice_opt, iteritems
from pygments import unistring as uni
from pygments.lexers.html import XmlLexer
class BooLexer(RegexLexer):
    """
    For `Boo <http://boo.codehaus.org/>`_ source code.
    """
    name = 'Boo'
    aliases = ['boo']
    filenames = ['*.boo']
    mimetypes = ['text/x-boo']
    tokens = {'root': [('\\s+', Text), ('(#|//).*$', Comment.Single), ('/[*]', Comment.Multiline, 'comment'), ('[]{}:(),.;[]', Punctuation), ('\\\\\\n', Text), ('\\\\', Text), ('(in|is|and|or|not)\\b', Operator.Word), ('/(\\\\\\\\|\\\\/|[^/\\s])/', String.Regex), ('@/(\\\\\\\\|\\\\/|[^/])*/', String.Regex), ('=~|!=|==|<<|>>|[-+/*%=<>&^|]', Operator), ('(as|abstract|callable|constructor|destructor|do|import|enum|event|final|get|interface|internal|of|override|partial|private|protected|public|return|set|static|struct|transient|virtual|yield|super|and|break|cast|continue|elif|else|ensure|except|for|given|goto|if|in|is|isa|not|or|otherwise|pass|raise|ref|try|unless|when|while|from|as)\\b', Keyword), ('def(?=\\s+\\(.*?\\))', Keyword), ('(def)(\\s+)', bygroups(Keyword, Text), 'funcname'), ('(class)(\\s+)', bygroups(Keyword, Text), 'classname'), ('(namespace)(\\s+)', bygroups(Keyword, Text), 'namespace'), ('(?<!\\.)(true|false|null|self|__eval__|__switch__|array|assert|checked|enumerate|filter|getter|len|lock|map|matrix|max|min|normalArrayIndexing|print|property|range|rawArrayIndexing|required|typeof|unchecked|using|yieldAll|zip)\\b', Name.Builtin), ('"""(\\\\\\\\|\\\\"|.*?)"""', String.Double), ('"(\\\\\\\\|\\\\"|[^"]*?)"', String.Double), ("'(\\\\\\\\|\\\\'|[^']*?)'", String.Single), ('[a-zA-Z_]\\w*', Name), ('(\\d+\\.\\d*|\\d*\\.\\d+)([fF][+-]?[0-9]+)?', Number.Float), ('[0-9][0-9.]*(ms?|d|h|s)', Number), ('0\\d+', Number.Oct), ('0x[a-fA-F0-9]+', Number.Hex), ('\\d+L', Number.Integer.Long), ('\\d+', Number.Integer)], 'comment': [('/[*]', Comment.Multiline, '#push'), ('[*]/', Comment.Multiline, '#pop'), ('[^/*]', Comment.Multiline), ('[*/]', Comment.Multiline)], 'funcname': [('[a-zA-Z_]\\w*', Name.Function, '#pop')], 'classname': [('[a-zA-Z_]\\w*', Name.Class, '#pop')], 'namespace': [('[a-zA-Z_][\\w.]*', Name.Namespace, '#pop')]}