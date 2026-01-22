import re
from pygments.lexer import Lexer, RegexLexer, include, bygroups, using, \
from pygments.util import get_bool_opt, shebang_matches
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments import unistring as uni
class DgLexer(RegexLexer):
    """
    Lexer for `dg <http://pyos.github.com/dg>`_,
    a functional and object-oriented programming language
    running on the CPython 3 VM.

    .. versionadded:: 1.6
    """
    name = 'dg'
    aliases = ['dg']
    filenames = ['*.dg']
    mimetypes = ['text/x-dg']
    tokens = {'root': [('\\s+', Text), ('#.*?$', Comment.Single), ('(?i)0b[01]+', Number.Bin), ('(?i)0o[0-7]+', Number.Oct), ('(?i)0x[0-9a-f]+', Number.Hex), ('(?i)[+-]?[0-9]+\\.[0-9]+(e[+-]?[0-9]+)?j?', Number.Float), ('(?i)[+-]?[0-9]+e[+-]?\\d+j?', Number.Float), ('(?i)[+-]?[0-9]+j?', Number.Integer), ("(?i)(br|r?b?)'''", String, combined('stringescape', 'tsqs', 'string')), ('(?i)(br|r?b?)"""', String, combined('stringescape', 'tdqs', 'string')), ("(?i)(br|r?b?)'", String, combined('stringescape', 'sqs', 'string')), ('(?i)(br|r?b?)"', String, combined('stringescape', 'dqs', 'string')), ("`\\w+'*`", Operator), ('\\b(and|in|is|or|where)\\b', Operator.Word), ('[!$%&*+\\-./:<-@\\\\^|~;,]+', Operator), (words(('bool', 'bytearray', 'bytes', 'classmethod', 'complex', 'dict', "dict'", 'float', 'frozenset', 'int', 'list', "list'", 'memoryview', 'object', 'property', 'range', 'set', "set'", 'slice', 'staticmethod', 'str', 'super', 'tuple', "tuple'", 'type'), prefix='(?<!\\.)', suffix="(?![\\'\\w])"), Name.Builtin), (words(('__import__', 'abs', 'all', 'any', 'bin', 'bind', 'chr', 'cmp', 'compile', 'complex', 'delattr', 'dir', 'divmod', 'drop', 'dropwhile', 'enumerate', 'eval', 'exhaust', 'filter', 'flip', 'foldl1?', 'format', 'fst', 'getattr', 'globals', 'hasattr', 'hash', 'head', 'hex', 'id', 'init', 'input', 'isinstance', 'issubclass', 'iter', 'iterate', 'last', 'len', 'locals', 'map', 'max', 'min', 'next', 'oct', 'open', 'ord', 'pow', 'print', 'repr', 'reversed', 'round', 'setattr', 'scanl1?', 'snd', 'sorted', 'sum', 'tail', 'take', 'takewhile', 'vars', 'zip'), prefix='(?<!\\.)', suffix="(?![\\'\\w])"), Name.Builtin), ("(?<!\\.)(self|Ellipsis|NotImplemented|None|True|False)(?!['\\w])", Name.Builtin.Pseudo), ("(?<!\\.)[A-Z]\\w*(Error|Exception|Warning)'*(?!['\\w])", Name.Exception), ("(?<!\\.)(Exception|GeneratorExit|KeyboardInterrupt|StopIteration|SystemExit)(?!['\\w])", Name.Exception), ("(?<![\\w.])(except|finally|for|if|import|not|otherwise|raise|subclass|while|with|yield)(?!['\\w])", Keyword.Reserved), ("[A-Z_]+'*(?!['\\w])", Name), ("[A-Z]\\w+'*(?!['\\w])", Keyword.Type), ("\\w+'*", Name), ('[()]', Punctuation), ('.', Error)], 'stringescape': [('\\\\([\\\\abfnrtv"\\\']|\\n|N\\{.*?\\}|u[a-fA-F0-9]{4}|U[a-fA-F0-9]{8}|x[a-fA-F0-9]{2}|[0-7]{1,3})', String.Escape)], 'string': [('%(\\(\\w+\\))?[-#0 +]*([0-9]+|[*])?(\\.([0-9]+|[*]))?[hlL]?[E-GXc-giorsux%]', String.Interpol), ('[^\\\\\\\'"%\\n]+', String), ('[\\\'"\\\\]', String), ('%', String), ('\\n', String)], 'dqs': [('"', String, '#pop')], 'sqs': [("'", String, '#pop')], 'tdqs': [('"""', String, '#pop')], 'tsqs': [("'''", String, '#pop')]}