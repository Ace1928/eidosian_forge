import re
from pygments.lexer import RegexLexer, include
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class ParaSailLexer(RegexLexer):
    """
    For `ParaSail <http://www.parasail-lang.org>`_ source code.

    .. versionadded:: 2.1
    """
    name = 'ParaSail'
    aliases = ['parasail']
    filenames = ['*.psi', '*.psl']
    mimetypes = ['text/x-parasail']
    flags = re.MULTILINE
    tokens = {'root': [('[^\\S\\n]+', Text), ('//.*?\\n', Comment.Single), ('\\b(and|or|xor)=', Operator.Word), ('\\b(and(\\s+then)?|or(\\s+else)?|xor|rem|mod|(is|not)\\s+null)\\b', Operator.Word), ('\\b(abs|abstract|all|block|class|concurrent|const|continue|each|end|exit|extends|exports|forward|func|global|implements|import|in|interface|is|lambda|locked|new|not|null|of|op|optional|private|queued|ref|return|reverse|separate|some|type|until|var|with|if|then|else|elsif|case|for|while|loop)\\b', Keyword.Reserved), ('(abstract\\s+)?(interface|class|op|func|type)', Keyword.Declaration), ('"[^"]*"', String), ('\\\\[\\\'ntrf"0]', String.Escape), ('#[a-zA-Z]\\w*', Literal), include('numbers'), ("'[^']'", String.Char), ('[a-zA-Z]\\w*', Name), ('(<==|==>|<=>|\\*\\*=|<\\|=|<<=|>>=|==|!=|=\\?|<=|>=|\\*\\*|<<|>>|=>|:=|\\+=|-=|\\*=|\\|=|\\||/=|\\+|-|\\*|/|\\.\\.|<\\.\\.|\\.\\.<|<\\.\\.<)', Operator), ('(<|>|\\[|\\]|\\(|\\)|\\||:|;|,|.|\\{|\\}|->)', Punctuation), ('\\n+', Text)], 'numbers': [('\\d[0-9_]*#[0-9a-fA-F][0-9a-fA-F_]*#', Number.Hex), ('0[xX][0-9a-fA-F][0-9a-fA-F_]*', Number.Hex), ('0[bB][01][01_]*', Number.Bin), ('\\d[0-9_]*\\.\\d[0-9_]*[eE][+-]\\d[0-9_]*', Number.Float), ('\\d[0-9_]*\\.\\d[0-9_]*', Number.Float), ('\\d[0-9_]*', Number.Integer)]}