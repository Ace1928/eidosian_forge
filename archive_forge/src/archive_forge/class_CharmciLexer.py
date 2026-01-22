import re
from pygments.lexer import RegexLexer, include, bygroups, inherit, words, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.c_cpp import CLexer, CppLexer
from pygments.lexers import _mql_builtins
class CharmciLexer(CppLexer):
    """
    For `Charm++ <https://charm.cs.illinois.edu>`_ interface files (.ci).

    .. versionadded:: 2.4
    """
    name = 'Charmci'
    aliases = ['charmci']
    filenames = ['*.ci']
    mimetypes = []
    tokens = {'keywords': [('(module)(\\s+)', bygroups(Keyword, Text), 'classname'), (words(('mainmodule', 'mainchare', 'chare', 'array', 'group', 'nodegroup', 'message', 'conditional')), Keyword), (words(('entry', 'aggregate', 'threaded', 'sync', 'exclusive', 'nokeep', 'notrace', 'immediate', 'expedited', 'inline', 'local', 'python', 'accel', 'readwrite', 'writeonly', 'accelblock', 'memcritical', 'packed', 'varsize', 'initproc', 'initnode', 'initcall', 'stacksize', 'createhere', 'createhome', 'reductiontarget', 'iget', 'nocopy', 'mutable', 'migratable', 'readonly')), Keyword), inherit]}