import re
from pygments.lexer import RegexLexer
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class X10Lexer(RegexLexer):
    """
    For the X10 language.

    .. versionadded:: 0.1
    """
    name = 'X10'
    aliases = ['x10', 'xten']
    filenames = ['*.x10']
    mimetypes = ['text/x-x10']
    keywords = ('as', 'assert', 'async', 'at', 'athome', 'ateach', 'atomic', 'break', 'case', 'catch', 'class', 'clocked', 'continue', 'def', 'default', 'do', 'else', 'final', 'finally', 'finish', 'for', 'goto', 'haszero', 'here', 'if', 'import', 'in', 'instanceof', 'interface', 'isref', 'new', 'offer', 'operator', 'package', 'return', 'struct', 'switch', 'throw', 'try', 'type', 'val', 'var', 'when', 'while')
    types = 'void'
    values = ('false', 'null', 'self', 'super', 'this', 'true')
    modifiers = ('abstract', 'extends', 'implements', 'native', 'offers', 'private', 'property', 'protected', 'public', 'static', 'throws', 'transient')
    tokens = {'root': [('[^\\S\\n]+', Text), ('//.*?\\n', Comment.Single), ('/\\*(.|\\n)*?\\*/', Comment.Multiline), ('\\b(%s)\\b' % '|'.join(keywords), Keyword), ('\\b(%s)\\b' % '|'.join(types), Keyword.Type), ('\\b(%s)\\b' % '|'.join(values), Keyword.Constant), ('\\b(%s)\\b' % '|'.join(modifiers), Keyword.Declaration), ('"(\\\\\\\\|\\\\"|[^"])*"', String), ("'\\\\.'|'[^\\\\]'|'\\\\u[0-9a-fA-F]{4}'", String.Char), ('.', Text)]}