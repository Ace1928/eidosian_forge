import re
from pygments.lexer import Lexer, RegexLexer, include, bygroups, using, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import shebang_matches
from pygments import unistring as uni
class KotlinLexer(RegexLexer):
    """
    For `Kotlin <http://kotlinlang.org/>`_
    source code.

    .. versionadded:: 1.5
    """
    name = 'Kotlin'
    aliases = ['kotlin']
    filenames = ['*.kt']
    mimetypes = ['text/x-kotlin']
    flags = re.MULTILINE | re.DOTALL | re.UNICODE
    kt_name = '@?[_' + uni.combine('Lu', 'Ll', 'Lt', 'Lm', 'Nl') + ']' + '[' + uni.combine('Lu', 'Ll', 'Lt', 'Lm', 'Nl', 'Nd', 'Pc', 'Cf', 'Mn', 'Mc') + ']*'
    kt_id = '(' + kt_name + '|`' + kt_name + '`)'
    tokens = {'root': [('^\\s*\\[.*?\\]', Name.Attribute), ('[^\\S\\n]+', Text), ('\\\\\\n', Text), ('//.*?\\n', Comment.Single), ('/[*].*?[*]/', Comment.Multiline), ('\\n', Text), ('::|!!|\\?[:.]', Operator), ('[~!%^&*()+=|\\[\\]:;,.<>/?-]', Punctuation), ('[{}]', Punctuation), ('@"(""|[^"])*"', String), ('"(\\\\\\\\|\\\\"|[^"\\n])*["\\n]', String), ("'\\\\.'|'[^\\\\]'", String.Char), ('[0-9](\\.[0-9]*)?([eE][+-][0-9]+)?[flFL]?|0[xX][0-9a-fA-F]+[Ll]?', Number), ('(class)(\\s+)(object)', bygroups(Keyword, Text, Keyword)), ('(class|interface|object)(\\s+)', bygroups(Keyword, Text), 'class'), ('(package|import)(\\s+)', bygroups(Keyword, Text), 'package'), ('(val|var)(\\s+)', bygroups(Keyword, Text), 'property'), ('(fun)(\\s+)', bygroups(Keyword, Text), 'function'), ('(abstract|annotation|as|break|by|catch|class|companion|const|constructor|continue|crossinline|data|do|dynamic|else|enum|external|false|final|finally|for|fun|get|if|import|in|infix|inline|inner|interface|internal|is|lateinit|noinline|null|object|open|operator|out|override|package|private|protected|public|reified|return|sealed|set|super|tailrec|this|throw|true|try|val|var|vararg|when|where|while)\\b', Keyword), (kt_id, Name)], 'package': [('\\S+', Name.Namespace, '#pop')], 'class': [(kt_id, Name.Class, '#pop')], 'property': [(kt_id, Name.Property, '#pop')], 'function': [(kt_id, Name.Function, '#pop')]}