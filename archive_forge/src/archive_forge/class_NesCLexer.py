import re
from pygments.lexer import RegexLexer, include, bygroups, inherit, words, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.c_cpp import CLexer, CppLexer
from pygments.lexers import _mql_builtins
class NesCLexer(CLexer):
    """
    For `nesC <https://github.com/tinyos/nesc>`_ source code with preprocessor
    directives.

    .. versionadded:: 2.0
    """
    name = 'nesC'
    aliases = ['nesc']
    filenames = ['*.nc']
    mimetypes = ['text/x-nescsrc']
    tokens = {'statements': [(words(('abstract', 'as', 'async', 'atomic', 'call', 'command', 'component', 'components', 'configuration', 'event', 'extends', 'generic', 'implementation', 'includes', 'interface', 'module', 'new', 'norace', 'post', 'provides', 'signal', 'task', 'uses'), suffix='\\b'), Keyword), (words(('nx_struct', 'nx_union', 'nx_int8_t', 'nx_int16_t', 'nx_int32_t', 'nx_int64_t', 'nx_uint8_t', 'nx_uint16_t', 'nx_uint32_t', 'nx_uint64_t'), suffix='\\b'), Keyword.Type), inherit]}