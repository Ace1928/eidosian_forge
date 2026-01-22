import re
from bisect import bisect
from pygments.lexer import RegexLexer, include, default, bygroups, using, this
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.python import PythonLexer
class AwkLexer(RegexLexer):
    """
    For Awk scripts.

    .. versionadded:: 1.5
    """
    name = 'Awk'
    aliases = ['awk', 'gawk', 'mawk', 'nawk']
    filenames = ['*.awk']
    mimetypes = ['application/x-awk']
    tokens = {'commentsandwhitespace': [('\\s+', Text), ('#.*$', Comment.Single)], 'slashstartsregex': [include('commentsandwhitespace'), ('/(\\\\.|[^[/\\\\\\n]|\\[(\\\\.|[^\\]\\\\\\n])*])+/\\B', String.Regex, '#pop'), ('(?=/)', Text, ('#pop', 'badregex')), default('#pop')], 'badregex': [('\\n', Text, '#pop')], 'root': [('^(?=\\s|/)', Text, 'slashstartsregex'), include('commentsandwhitespace'), ('\\+\\+|--|\\|\\||&&|in\\b|\\$|!?~|(\\*\\*|[-<>+*%\\^/!=|])=?', Operator, 'slashstartsregex'), ('[{(\\[;,]', Punctuation, 'slashstartsregex'), ('[})\\].]', Punctuation), ('(break|continue|do|while|exit|for|if|else|return)\\b', Keyword, 'slashstartsregex'), ('function\\b', Keyword.Declaration, 'slashstartsregex'), ('(atan2|cos|exp|int|log|rand|sin|sqrt|srand|gensub|gsub|index|length|match|split|sprintf|sub|substr|tolower|toupper|close|fflush|getline|next|nextfile|print|printf|strftime|systime|delete|system)\\b', Keyword.Reserved), ('(ARGC|ARGIND|ARGV|BEGIN|CONVFMT|ENVIRON|END|ERRNO|FIELDWIDTHS|FILENAME|FNR|FS|IGNORECASE|NF|NR|OFMT|OFS|ORFS|RLENGTH|RS|RSTART|RT|SUBSEP)\\b', Name.Builtin), ('[$a-zA-Z_]\\w*', Name.Other), ('[0-9][0-9]*\\.[0-9]+([eE][0-9]+)?[fd]?', Number.Float), ('0x[0-9a-fA-F]+', Number.Hex), ('[0-9]+', Number.Integer), ('"(\\\\\\\\|\\\\"|[^"])*"', String.Double), ("'(\\\\\\\\|\\\\'|[^'])*'", String.Single)]}