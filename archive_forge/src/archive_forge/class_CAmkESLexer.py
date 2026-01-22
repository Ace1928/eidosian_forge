from pygments.lexer import RegexLexer, include, words
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class CAmkESLexer(RegexLexer):
    """
    Basic lexer for the input language for the
    `CAmkES <https://sel4.systems/CAmkES/>`_ component platform.

    .. versionadded:: 2.1
    """
    name = 'CAmkES'
    aliases = ['camkes', 'idl4']
    filenames = ['*.camkes', '*.idl4']
    tokens = {'root': [('^\\s*#.*\\n', Comment.Preproc), ('\\s+', Text), ('/\\*(.|\\n)*?\\*/', Comment), ('//.*\\n', Comment), ('[\\[(){},.;\\]]', Punctuation), ('[~!%^&*+=|?:<>/-]', Operator), (words(('assembly', 'attribute', 'component', 'composition', 'configuration', 'connection', 'connector', 'consumes', 'control', 'dataport', 'Dataport', 'Dataports', 'emits', 'event', 'Event', 'Events', 'export', 'from', 'group', 'hardware', 'has', 'interface', 'Interface', 'maybe', 'procedure', 'Procedure', 'Procedures', 'provides', 'template', 'thread', 'threads', 'to', 'uses', 'with'), suffix='\\b'), Keyword), (words(('bool', 'boolean', 'Buf', 'char', 'character', 'double', 'float', 'in', 'inout', 'int', 'int16_6', 'int32_t', 'int64_t', 'int8_t', 'integer', 'mutex', 'out', 'real', 'refin', 'semaphore', 'signed', 'string', 'struct', 'uint16_t', 'uint32_t', 'uint64_t', 'uint8_t', 'uintptr_t', 'unsigned', 'void'), suffix='\\b'), Keyword.Type), ('[a-zA-Z_]\\w*_(priority|domain|buffer)', Keyword.Reserved), (words(('dma_pool', 'from_access', 'to_access'), suffix='\\b'), Keyword.Reserved), ('import\\s+(<[^>]*>|"[^"]*");', Comment.Preproc), ('include\\s+(<[^>]*>|"[^"]*");', Comment.Preproc), ('0[xX][\\da-fA-F]+', Number.Hex), ('-?[\\d]+', Number), ('-?[\\d]+\\.[\\d]+', Number.Float), ('"[^"]*"', String), ('[Tt]rue|[Ff]alse', Name.Builtin), ('[a-zA-Z_]\\w*', Name)]}