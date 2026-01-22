import re
from pygments.lexer import Lexer, RegexLexer, include, bygroups, using, \
from pygments.util import get_bool_opt, shebang_matches
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments import unistring as uni
class Python3Lexer(RegexLexer):
    """
    For `Python <http://www.python.org>`_ source code (version 3.0).

    .. versionadded:: 0.10
    """
    name = 'Python 3'
    aliases = ['python3', 'py3']
    filenames = []
    mimetypes = ['text/x-python3', 'application/x-python3']
    flags = re.MULTILINE | re.UNICODE
    uni_name = '[%s][%s]*' % (uni.xid_start, uni.xid_continue)

    def innerstring_rules(ttype):
        return [('%(\\(\\w+\\))?[-#0 +]*([0-9]+|[*])?(\\.([0-9]+|[*]))?[hlL]?[E-GXc-giorsux%]', String.Interpol), ('\\{((\\w+)((\\.\\w+)|(\\[[^\\]]+\\]))*)?(\\![sra])?(\\:(.?[<>=\\^])?[-+ ]?#?0?(\\d+)?,?(\\.\\d+)?[E-GXb-gnosx%]?)?\\}', String.Interpol), ('[^\\\\\\\'"%{\\n]+', ttype), ('[\\\'"\\\\]', ttype), ('%|(\\{{1,2})', ttype)]
    tokens = PythonLexer.tokens.copy()
    tokens['keywords'] = [(words(('assert', 'async', 'await', 'break', 'continue', 'del', 'elif', 'else', 'except', 'finally', 'for', 'global', 'if', 'lambda', 'pass', 'raise', 'nonlocal', 'return', 'try', 'while', 'yield', 'yield from', 'as', 'with'), suffix='\\b'), Keyword), (words(('True', 'False', 'None'), suffix='\\b'), Keyword.Constant)]
    tokens['builtins'] = [(words(('__import__', 'abs', 'all', 'any', 'bin', 'bool', 'bytearray', 'bytes', 'chr', 'classmethod', 'cmp', 'compile', 'complex', 'delattr', 'dict', 'dir', 'divmod', 'enumerate', 'eval', 'filter', 'float', 'format', 'frozenset', 'getattr', 'globals', 'hasattr', 'hash', 'hex', 'id', 'input', 'int', 'isinstance', 'issubclass', 'iter', 'len', 'list', 'locals', 'map', 'max', 'memoryview', 'min', 'next', 'object', 'oct', 'open', 'ord', 'pow', 'print', 'property', 'range', 'repr', 'reversed', 'round', 'set', 'setattr', 'slice', 'sorted', 'staticmethod', 'str', 'sum', 'super', 'tuple', 'type', 'vars', 'zip'), prefix='(?<!\\.)', suffix='\\b'), Name.Builtin), ('(?<!\\.)(self|Ellipsis|NotImplemented|cls)\\b', Name.Builtin.Pseudo), (words(('ArithmeticError', 'AssertionError', 'AttributeError', 'BaseException', 'BufferError', 'BytesWarning', 'DeprecationWarning', 'EOFError', 'EnvironmentError', 'Exception', 'FloatingPointError', 'FutureWarning', 'GeneratorExit', 'IOError', 'ImportError', 'ImportWarning', 'IndentationError', 'IndexError', 'KeyError', 'KeyboardInterrupt', 'LookupError', 'MemoryError', 'NameError', 'NotImplementedError', 'OSError', 'OverflowError', 'PendingDeprecationWarning', 'ReferenceError', 'ResourceWarning', 'RuntimeError', 'RuntimeWarning', 'StopIteration', 'SyntaxError', 'SyntaxWarning', 'SystemError', 'SystemExit', 'TabError', 'TypeError', 'UnboundLocalError', 'UnicodeDecodeError', 'UnicodeEncodeError', 'UnicodeError', 'UnicodeTranslateError', 'UnicodeWarning', 'UserWarning', 'ValueError', 'VMSError', 'Warning', 'WindowsError', 'ZeroDivisionError', 'BlockingIOError', 'ChildProcessError', 'ConnectionError', 'BrokenPipeError', 'ConnectionAbortedError', 'ConnectionRefusedError', 'ConnectionResetError', 'FileExistsError', 'FileNotFoundError', 'InterruptedError', 'IsADirectoryError', 'NotADirectoryError', 'PermissionError', 'ProcessLookupError', 'TimeoutError'), prefix='(?<!\\.)', suffix='\\b'), Name.Exception)]
    tokens['magicfuncs'] = [(words(('__abs__', '__add__', '__aenter__', '__aexit__', '__aiter__', '__and__', '__anext__', '__await__', '__bool__', '__bytes__', '__call__', '__complex__', '__contains__', '__del__', '__delattr__', '__delete__', '__delitem__', '__dir__', '__divmod__', '__enter__', '__eq__', '__exit__', '__float__', '__floordiv__', '__format__', '__ge__', '__get__', '__getattr__', '__getattribute__', '__getitem__', '__gt__', '__hash__', '__iadd__', '__iand__', '__ifloordiv__', '__ilshift__', '__imatmul__', '__imod__', '__import__', '__imul__', '__index__', '__init__', '__instancecheck__', '__int__', '__invert__', '__ior__', '__ipow__', '__irshift__', '__isub__', '__iter__', '__itruediv__', '__ixor__', '__le__', '__len__', '__length_hint__', '__lshift__', '__lt__', '__matmul__', '__missing__', '__mod__', '__mul__', '__ne__', '__neg__', '__new__', '__next__', '__or__', '__pos__', '__pow__', '__prepare__', '__radd__', '__rand__', '__rdivmod__', '__repr__', '__reversed__', '__rfloordiv__', '__rlshift__', '__rmatmul__', '__rmod__', '__rmul__', '__ror__', '__round__', '__rpow__', '__rrshift__', '__rshift__', '__rsub__', '__rtruediv__', '__rxor__', '__set__', '__setattr__', '__setitem__', '__str__', '__sub__', '__subclasscheck__', '__truediv__', '__xor__'), suffix='\\b'), Name.Function.Magic)]
    tokens['magicvars'] = [(words(('__annotations__', '__bases__', '__class__', '__closure__', '__code__', '__defaults__', '__dict__', '__doc__', '__file__', '__func__', '__globals__', '__kwdefaults__', '__module__', '__mro__', '__name__', '__objclass__', '__qualname__', '__self__', '__slots__', '__weakref__'), suffix='\\b'), Name.Variable.Magic)]
    tokens['numbers'] = [('(\\d+\\.\\d*|\\d*\\.\\d+)([eE][+-]?[0-9]+)?', Number.Float), ('\\d+[eE][+-]?[0-9]+j?', Number.Float), ('0[oO][0-7]+', Number.Oct), ('0[bB][01]+', Number.Bin), ('0[xX][a-fA-F0-9]+', Number.Hex), ('\\d+', Number.Integer)]
    tokens['backtick'] = []
    tokens['name'] = [('@\\w+', Name.Decorator), ('@', Operator), (uni_name, Name)]
    tokens['funcname'] = [(uni_name, Name.Function, '#pop')]
    tokens['classname'] = [(uni_name, Name.Class, '#pop')]
    tokens['import'] = [('(\\s+)(as)(\\s+)', bygroups(Text, Keyword, Text)), ('\\.', Name.Namespace), (uni_name, Name.Namespace), ('(\\s*)(,)(\\s*)', bygroups(Text, Operator, Text)), default('#pop')]
    tokens['fromimport'] = [('(\\s+)(import)\\b', bygroups(Text, Keyword), '#pop'), ('\\.', Name.Namespace), (uni_name, Name.Namespace), default('#pop')]
    tokens['strings-single'] = innerstring_rules(String.Single)
    tokens['strings-double'] = innerstring_rules(String.Double)

    def analyse_text(text):
        return shebang_matches(text, 'pythonw?3(\\.\\d)?')