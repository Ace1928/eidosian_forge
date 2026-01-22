import re
from pygments.lexer import RegexLexer, include, bygroups, default, combined, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import get_bool_opt, get_list_opt, iteritems
class MOOCodeLexer(RegexLexer):
    """
    For `MOOCode <http://www.moo.mud.org/>`_ (the MOO scripting
    language).

    .. versionadded:: 0.9
    """
    name = 'MOOCode'
    filenames = ['*.moo']
    aliases = ['moocode', 'moo']
    mimetypes = ['text/x-moocode']
    tokens = {'root': [('(0|[1-9][0-9_]*)', Number.Integer), ('"(\\\\\\\\|\\\\"|[^"])*"', String), ('(E_PERM|E_DIV)', Name.Exception), ('((#[-0-9]+)|(\\$\\w+))', Name.Entity), ('\\b(if|else|elseif|endif|for|endfor|fork|endfork|while|endwhile|break|continue|return|try|except|endtry|finally|in)\\b', Keyword), ('(random|length)', Name.Builtin), ('(player|caller|this|args)', Name.Variable.Instance), ('\\s+', Text), ('\\n', Text), ('([!;=,{}&|:.\\[\\]@()<>?]+)', Operator), ('(\\w+)(\\()', bygroups(Name.Function, Operator)), ('(\\w+)', Text)]}