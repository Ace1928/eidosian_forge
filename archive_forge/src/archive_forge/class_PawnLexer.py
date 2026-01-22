from pygments.lexer import RegexLexer
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import get_bool_opt
class PawnLexer(RegexLexer):
    """
    For Pawn source code.

    .. versionadded:: 2.0
    """
    name = 'Pawn'
    aliases = ['pawn']
    filenames = ['*.p', '*.pwn', '*.inc']
    mimetypes = ['text/x-pawn']
    _ws = '(?:\\s|//.*?\\n|/[*][\\w\\W]*?[*]/)+'
    _ws1 = '\\s*(?:/[*].*?[*]/\\s*)*'
    tokens = {'root': [('^#if\\s+0', Comment.Preproc, 'if0'), ('^#', Comment.Preproc, 'macro'), ('^' + _ws1 + '#if\\s+0', Comment.Preproc, 'if0'), ('^' + _ws1 + '#', Comment.Preproc, 'macro'), ('\\n', Text), ('\\s+', Text), ('\\\\\\n', Text), ('/(\\\\\\n)?/(\\n|(.|\\n)*?[^\\\\]\\n)', Comment.Single), ('/(\\\\\\n)?\\*[\\w\\W]*?\\*(\\\\\\n)?/', Comment.Multiline), ('[{}]', Punctuation), ('L?"', String, 'string'), ("L?'(\\\\.|\\\\[0-7]{1,3}|\\\\x[a-fA-F0-9]{1,2}|[^\\\\\\'\\n])'", String.Char), ('(\\d+\\.\\d*|\\.\\d+|\\d+)[eE][+-]?\\d+[LlUu]*', Number.Float), ('(\\d+\\.\\d*|\\.\\d+|\\d+[fF])[fF]?', Number.Float), ('0x[0-9a-fA-F]+[LlUu]*', Number.Hex), ('0[0-7]+[LlUu]*', Number.Oct), ('\\d+[LlUu]*', Number.Integer), ('\\*/', Error), ('[~!%^&*+=|?:<>/-]', Operator), ('[()\\[\\],.;]', Punctuation), ('(switch|case|default|const|new|static|char|continue|break|if|else|for|while|do|operator|enum|public|return|sizeof|tagof|state|goto)\\b', Keyword), ('(bool|Float)\\b', Keyword.Type), ('(true|false)\\b', Keyword.Constant), ('[a-zA-Z_]\\w*', Name)], 'string': [('"', String, '#pop'), ('\\\\([\\\\abfnrtv"\\\']|x[a-fA-F0-9]{2,4}|[0-7]{1,3})', String.Escape), ('[^\\\\"\\n]+', String), ('\\\\\\n', String), ('\\\\', String)], 'macro': [('[^/\\n]+', Comment.Preproc), ('/\\*(.|\\n)*?\\*/', Comment.Multiline), ('//.*?\\n', Comment.Single, '#pop'), ('/', Comment.Preproc), ('(?<=\\\\)\\n', Comment.Preproc), ('\\n', Comment.Preproc, '#pop')], 'if0': [('^\\s*#if.*?(?<!\\\\)\\n', Comment.Preproc, '#push'), ('^\\s*#endif.*?(?<!\\\\)\\n', Comment.Preproc, '#pop'), ('.*?\\n', Comment)]}