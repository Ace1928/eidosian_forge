import re
from pygments.lexer import Lexer, RegexLexer, do_insertions, bygroups, \
from pygments.token import Punctuation, \
from pygments.util import shebang_matches
class FishShellLexer(RegexLexer):
    """
    Lexer for Fish shell scripts.

    .. versionadded:: 2.1
    """
    name = 'Fish'
    aliases = ['fish', 'fishshell']
    filenames = ['*.fish', '*.load']
    mimetypes = ['application/x-fish']
    tokens = {'root': [include('basic'), include('data'), include('interp')], 'interp': [('\\$\\(\\(', Keyword, 'math'), ('\\(', Keyword, 'paren'), ('\\$#?(\\w+|.)', Name.Variable)], 'basic': [('\\b(begin|end|if|else|while|break|for|in|return|function|block|case|continue|switch|not|and|or|set|echo|exit|pwd|true|false|cd|count|test)(\\s*)\\b', bygroups(Keyword, Text)), ('\\b(alias|bg|bind|breakpoint|builtin|command|commandline|complete|contains|dirh|dirs|emit|eval|exec|fg|fish|fish_config|fish_indent|fish_pager|fish_prompt|fish_right_prompt|fish_update_completions|fishd|funced|funcsave|functions|help|history|isatty|jobs|math|mimedb|nextd|open|popd|prevd|psub|pushd|random|read|set_color|source|status|trap|type|ulimit|umask|vared|fc|getopts|hash|kill|printf|time|wait)\\s*\\b(?!\\.)', Name.Builtin), ('#.*\\n', Comment), ('\\\\[\\w\\W]', String.Escape), ('(\\b\\w+)(\\s*)(=)', bygroups(Name.Variable, Text, Operator)), ('[\\[\\]()=]', Operator), ("<<-?\\s*(\\'?)\\\\?(\\w+)[\\w\\W]+?\\2", String)], 'data': [('(?s)\\$?"(\\\\\\\\|\\\\[0-7]+|\\\\.|[^"\\\\$])*"', String.Double), ('"', String.Double, 'string'), ("(?s)\\$'(\\\\\\\\|\\\\[0-7]+|\\\\.|[^'\\\\])*'", String.Single), ("(?s)'.*?'", String.Single), (';', Punctuation), ('&|\\||\\^|<|>', Operator), ('\\s+', Text), ('\\d+(?= |\\Z)', Number), ('[^=\\s\\[\\]{}()$"\\\'`\\\\<&|;]+', Text)], 'string': [('"', String.Double, '#pop'), ('(?s)(\\\\\\\\|\\\\[0-7]+|\\\\.|[^"\\\\$])+', String.Double), include('interp')], 'paren': [('\\)', Keyword, '#pop'), include('root')], 'math': [('\\)\\)', Keyword, '#pop'), ('[-+*/%^|&]|\\*\\*|\\|\\|', Operator), ('\\d+#\\d+', Number), ('\\d+#(?! )', Number), ('\\d+', Number), include('root')]}