import re
from pygments.lexer import ExtendedRegexLexer, RegexLexer, bygroups, words, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class ProtoBufLexer(RegexLexer):
    """
    Lexer for `Protocol Buffer <http://code.google.com/p/protobuf/>`_
    definition files.

    .. versionadded:: 1.4
    """
    name = 'Protocol Buffer'
    aliases = ['protobuf', 'proto']
    filenames = ['*.proto']
    tokens = {'root': [('[ \\t]+', Text), ('[,;{}\\[\\]()<>]', Punctuation), ('/(\\\\\\n)?/(\\n|(.|\\n)*?[^\\\\]\\n)', Comment.Single), ('/(\\\\\\n)?\\*(.|\\n)*?\\*(\\\\\\n)?/', Comment.Multiline), (words(('import', 'option', 'optional', 'required', 'repeated', 'default', 'packed', 'ctype', 'extensions', 'to', 'max', 'rpc', 'returns', 'oneof'), prefix='\\b', suffix='\\b'), Keyword), (words(('int32', 'int64', 'uint32', 'uint64', 'sint32', 'sint64', 'fixed32', 'fixed64', 'sfixed32', 'sfixed64', 'float', 'double', 'bool', 'string', 'bytes'), suffix='\\b'), Keyword.Type), ('(true|false)\\b', Keyword.Constant), ('(package)(\\s+)', bygroups(Keyword.Namespace, Text), 'package'), ('(message|extend)(\\s+)', bygroups(Keyword.Declaration, Text), 'message'), ('(enum|group|service)(\\s+)', bygroups(Keyword.Declaration, Text), 'type'), ('\\".*?\\"', String), ("\\'.*?\\'", String), ('(\\d+\\.\\d*|\\.\\d+|\\d+)[eE][+-]?\\d+[LlUu]*', Number.Float), ('(\\d+\\.\\d*|\\.\\d+|\\d+[fF])[fF]?', Number.Float), ('(\\-?(inf|nan))\\b', Number.Float), ('0x[0-9a-fA-F]+[LlUu]*', Number.Hex), ('0[0-7]+[LlUu]*', Number.Oct), ('\\d+[LlUu]*', Number.Integer), ('[+-=]', Operator), ('([a-zA-Z_][\\w.]*)([ \\t]*)(=)', bygroups(Name.Attribute, Text, Operator)), ('[a-zA-Z_][\\w.]*', Name)], 'package': [('[a-zA-Z_]\\w*', Name.Namespace, '#pop'), default('#pop')], 'message': [('[a-zA-Z_]\\w*', Name.Class, '#pop'), default('#pop')], 'type': [('[a-zA-Z_]\\w*', Name, '#pop'), default('#pop')]}