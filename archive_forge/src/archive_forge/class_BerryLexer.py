from pygments.lexer import RegexLexer, words, include, bygroups
from pygments.token import Comment, Whitespace, Operator, Keyword, Name, \
class BerryLexer(RegexLexer):
    """
    For `berry <http://github.com/berry-lang/berry>`_ source code.

    .. versionadded:: 2.12.0
    """
    name = 'Berry'
    aliases = ['berry', 'be']
    filenames = ['*.be']
    mimetypes = ['text/x-berry', 'application/x-berry']
    _name = '\\b[^\\W\\d]\\w*'
    tokens = {'root': [include('whitespace'), include('numbers'), include('keywords'), (f'(def)(\\s+)({_name})', bygroups(Keyword.Declaration, Whitespace, Name.Function)), (f'\\b(class)(\\s+)({_name})', bygroups(Keyword.Declaration, Whitespace, Name.Class)), (f'\\b(import)(\\s+)({_name})', bygroups(Keyword.Namespace, Whitespace, Name.Namespace)), include('expr')], 'expr': [('[^\\S\\n]+', Whitespace), ('\\.\\.|[~!%^&*+=|?:<>/-]', Operator), ('[(){}\\[\\],.;]', Punctuation), include('controls'), include('builtins'), include('funccall'), include('member'), include('name'), include('strings')], 'whitespace': [('\\s+', Whitespace), ('#-(.|\\n)*?-#', Comment.Multiline), ('#.*?$', Comment.Single)], 'keywords': [(words(('as', 'break', 'continue', 'import', 'static', 'self', 'super'), suffix='\\b'), Keyword.Reserved), ('(true|false|nil)\\b', Keyword.Constant), ('(var|def)\\b', Keyword.Declaration)], 'controls': [(words(('if', 'elif', 'else', 'for', 'while', 'do', 'end', 'break', 'continue', 'return', 'try', 'except', 'raise'), suffix='\\b'), Keyword)], 'builtins': [(words(('assert', 'bool', 'input', 'classname', 'classof', 'number', 'real', 'bytes', 'compile', 'map', 'list', 'int', 'isinstance', 'print', 'range', 'str', 'super', 'module', 'size', 'issubclass', 'open', 'file', 'type', 'call'), suffix='\\b'), Name.Builtin)], 'numbers': [('0[xX][a-fA-F0-9]+', Number.Hex), ('-?\\d+', Number.Integer), ('(-?\\d+\\.?|\\.\\d)\\d*([eE][+-]?\\d+)?', Number.Float)], 'name': [(_name, Name)], 'funccall': [(f'{_name}(?=\\s*\\()', Name.Function, '#pop')], 'member': [(f'(?<=\\.){_name}\\b(?!\\()', Name.Attribute, '#pop')], 'strings': [('"([^\\\\]|\\\\.)*?"', String.Double, '#pop'), ("\\'([^\\\\]|\\\\.)*?\\'", String.Single, '#pop')]}