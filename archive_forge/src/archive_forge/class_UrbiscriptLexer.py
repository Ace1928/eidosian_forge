import re
from pygments.lexer import ExtendedRegexLexer, words
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class UrbiscriptLexer(ExtendedRegexLexer):
    """
    For UrbiScript source code.

    .. versionadded:: 1.5
    """
    name = 'UrbiScript'
    aliases = ['urbiscript']
    filenames = ['*.u']
    mimetypes = ['application/x-urbiscript']
    flags = re.DOTALL

    def blob_callback(lexer, match, ctx):
        text_before_blob = match.group(1)
        blob_start = match.group(2)
        blob_size_str = match.group(3)
        blob_size = int(blob_size_str)
        yield (match.start(), String, text_before_blob)
        ctx.pos += len(text_before_blob)
        if ctx.text[match.end() + blob_size] != ')':
            result = '\\B(' + blob_size_str + ')('
            yield (match.start(), String, result)
            ctx.pos += len(result)
            return
        blob_text = blob_start + ctx.text[match.end():match.end() + blob_size] + ')'
        yield (match.start(), String.Escape, blob_text)
        ctx.pos = match.end() + blob_size + 1
    tokens = {'root': [('\\s+', Text), ('//.*?\\n', Comment), ('/\\*', Comment.Multiline, 'comment'), ('(every|for|loop|while)(?:;|&|\\||,)', Keyword), (words(('assert', 'at', 'break', 'case', 'catch', 'closure', 'compl', 'continue', 'default', 'else', 'enum', 'every', 'external', 'finally', 'for', 'freezeif', 'if', 'new', 'onleave', 'return', 'stopif', 'switch', 'this', 'throw', 'timeout', 'try', 'waituntil', 'whenever', 'while'), suffix='\\b'), Keyword), (words(('asm', 'auto', 'bool', 'char', 'const_cast', 'delete', 'double', 'dynamic_cast', 'explicit', 'export', 'extern', 'float', 'friend', 'goto', 'inline', 'int', 'long', 'mutable', 'namespace', 'register', 'reinterpret_cast', 'short', 'signed', 'sizeof', 'static_cast', 'struct', 'template', 'typedef', 'typeid', 'typename', 'union', 'unsigned', 'using', 'virtual', 'volatile', 'wchar_t'), suffix='\\b'), Keyword.Reserved), ('(emit|foreach|internal|loopn|static)\\b', Keyword), ('(private|protected|public)\\b', Keyword), ('(var|do|const|function|class)\\b', Keyword.Declaration), ('(true|false|nil|void)\\b', Keyword.Constant), (words(('Barrier', 'Binary', 'Boolean', 'CallMessage', 'Channel', 'Code', 'Comparable', 'Container', 'Control', 'Date', 'Dictionary', 'Directory', 'Duration', 'Enumeration', 'Event', 'Exception', 'Executable', 'File', 'Finalizable', 'Float', 'FormatInfo', 'Formatter', 'Global', 'Group', 'Hash', 'InputStream', 'IoService', 'Job', 'Kernel', 'Lazy', 'List', 'Loadable', 'Lobby', 'Location', 'Logger', 'Math', 'Mutex', 'nil', 'Object', 'Orderable', 'OutputStream', 'Pair', 'Path', 'Pattern', 'Position', 'Primitive', 'Process', 'Profile', 'PseudoLazy', 'PubSub', 'RangeIterable', 'Regexp', 'Semaphore', 'Server', 'Singleton', 'Socket', 'StackFrame', 'Stream', 'String', 'System', 'Tag', 'Timeout', 'Traceable', 'TrajectoryGenerator', 'Triplet', 'Tuple', 'UObject', 'UValue', 'UVar'), suffix='\\b'), Name.Builtin), ('(?:this)\\b', Name.Builtin.Pseudo), ('(?:[-=+*%/<>~^:]+|\\.&?|\\|\\||&&)', Operator), ('(?:and_eq|and|bitand|bitor|in|not|not_eq|or_eq|or|xor_eq|xor)\\b', Operator.Word), ('[{}\\[\\]()]+', Punctuation), ('(?:;|\\||,|&|\\?|!)+', Punctuation), ('[$a-zA-Z_]\\w*', Name.Other), ('0x[0-9a-fA-F]+', Number.Hex), ('(?:[0-9]+(?:(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)?((?:rad|deg|grad)|(?:ms|s|min|h|d))?)\\b', Number.Float), ('"', String.Double, 'string.double'), ("'", String.Single, 'string.single')], 'string.double': [('((?:\\\\\\\\|\\\\"|[^"])*?)(\\\\B\\((\\d+)\\)\\()', blob_callback), ('(\\\\\\\\|\\\\"|[^"])*?"', String.Double, '#pop')], 'string.single': [("((?:\\\\\\\\|\\\\'|[^'])*?)(\\\\B\\((\\d+)\\)\\()", blob_callback), ("(\\\\\\\\|\\\\'|[^'])*?'", String.Single, '#pop')], 'comment': [('[^*/]', Comment.Multiline), ('/\\*', Comment.Multiline, '#push'), ('\\*/', Comment.Multiline, '#pop'), ('[*/]', Comment.Multiline)]}