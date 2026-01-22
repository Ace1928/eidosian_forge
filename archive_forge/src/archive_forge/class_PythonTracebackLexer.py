import re
from pygments.lexer import Lexer, RegexLexer, include, bygroups, using, \
from pygments.util import get_bool_opt, shebang_matches
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments import unistring as uni
class PythonTracebackLexer(RegexLexer):
    """
    For Python tracebacks.

    .. versionadded:: 0.7
    """
    name = 'Python Traceback'
    aliases = ['pytb']
    filenames = ['*.pytb']
    mimetypes = ['text/x-python-traceback']
    tokens = {'root': [('^(\\^C)?(Traceback.*\\n)', bygroups(Text, Generic.Traceback), 'intb'), ('^(?=  File "[^"]+", line \\d+)', Generic.Traceback, 'intb'), ('^.*\\n', Other)], 'intb': [('^(  File )("[^"]+")(, line )(\\d+)(, in )(.+)(\\n)', bygroups(Text, Name.Builtin, Text, Number, Text, Name, Text)), ('^(  File )("[^"]+")(, line )(\\d+)(\\n)', bygroups(Text, Name.Builtin, Text, Number, Text)), ('^(    )(.+)(\\n)', bygroups(Text, using(PythonLexer), Text)), ('^([ \\t]*)(\\.\\.\\.)(\\n)', bygroups(Text, Comment, Text)), ('^([^:]+)(: )(.+)(\\n)', bygroups(Generic.Error, Text, Name, Text), '#pop'), ('^([a-zA-Z_]\\w*)(:?\\n)', bygroups(Generic.Error, Text), '#pop')]}