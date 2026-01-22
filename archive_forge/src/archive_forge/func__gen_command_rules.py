from pygments.lexer import RegexLexer, include, words
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import shebang_matches
def _gen_command_rules(keyword_cmds_re, builtin_cmds_re, context=''):
    return [(keyword_cmds_re, Keyword, 'params' + context), (builtin_cmds_re, Name.Builtin, 'params' + context), ('([\\w.-]+)', Name.Variable, 'params' + context), ('#', Comment, 'comment')]