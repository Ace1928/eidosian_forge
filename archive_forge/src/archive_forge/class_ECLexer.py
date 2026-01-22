import re
from pygments.lexer import RegexLexer, include, bygroups, inherit, words, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.c_cpp import CLexer, CppLexer
from pygments.lexers import _mql_builtins
class ECLexer(CLexer):
    """
    For eC source code with preprocessor directives.

    .. versionadded:: 1.5
    """
    name = 'eC'
    aliases = ['ec']
    filenames = ['*.ec', '*.eh']
    mimetypes = ['text/x-echdr', 'text/x-ecsrc']
    tokens = {'statements': [(words(('virtual', 'class', 'private', 'public', 'property', 'import', 'delete', 'new', 'new0', 'renew', 'renew0', 'define', 'get', 'set', 'remote', 'dllexport', 'dllimport', 'stdcall', 'subclass', '__on_register_module', 'namespace', 'using', 'typed_object', 'any_object', 'incref', 'register', 'watch', 'stopwatching', 'firewatchers', 'watchable', 'class_designer', 'class_fixed', 'class_no_expansion', 'isset', 'class_default_property', 'property_category', 'class_data', 'class_property', 'thisclass', 'dbtable', 'dbindex', 'database_open', 'dbfield'), suffix='\\b'), Keyword), (words(('uint', 'uint16', 'uint32', 'uint64', 'bool', 'byte', 'unichar', 'int64'), suffix='\\b'), Keyword.Type), ('(class)(\\s+)', bygroups(Keyword, Text), 'classname'), ('(null|value|this)\\b', Name.Builtin), inherit], 'classname': [('[a-zA-Z_]\\w*', Name.Class, '#pop'), ('\\s*(?=>)', Text, '#pop')]}