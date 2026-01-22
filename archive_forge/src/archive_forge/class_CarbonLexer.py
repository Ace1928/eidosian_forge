import re
from pygments.lexer import RegexLexer, bygroups, words
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class CarbonLexer(RegexLexer):
    """
    For Carbon source.

    .. versionadded:: 2.15
    """
    name = 'Carbon'
    url = 'https://github.com/carbon-language/carbon-lang'
    filenames = ['*.carbon']
    aliases = ['carbon']
    mimetypes = ['text/x-carbon']
    flags = re.MULTILINE | re.DOTALL
    tokens = {'root': [('\\n', Whitespace), ('\\s+', Whitespace), ('\\\\\\n', Text), ('//(.*?)\\n', Comment.Single), ('/(\\\\\\n)?[*].*?[*](\\\\\\n)?/', Comment.Multiline), ('(package|import|api|namespace|library)\\b', Keyword.Namespace), ('(abstract|alias|fn|class|interface|let|var|virtual|external|base|addr|extends|choice|constraint|impl)\\b', Keyword.Declaration), (words(('as', 'or', 'not', 'and', 'break', 'continue', 'case', 'default', 'if', 'else', 'destructor', 'for', 'forall', 'while', 'where', 'then', 'in', 'is', 'return', 'returned', 'friend', 'partial', 'private', 'protected', 'observe', 'Self', 'override', 'final', 'match', 'type', 'like'), suffix='\\b'), Keyword), ('(self)\\b', Keyword.Pseudo), ('(true|false)\\b', Keyword.Constant), ('(auto|bool|string|i8|i16|i32|i64|u8|u16|u32|u64|f8|f16|f32|f64)\\b', Keyword.Type), ('[0-9]*[.][0-9]+', Number.Double), ('0b[01]+', Number.Bin), ('0o[0-7]+', Number.Oct), ('0x[0-9a-fA-F]+', Number.Hex), ('[0-9]+', Number.Integer), ('"(\\\\.|[^"\\\\])*"', String), ("\\'(\\\\.|[^\\'\\\\])\\'", String.Char), ('<<=|>>=|<<|>>|<=|>=|\\+=|-=|\\*=|/=|\\%=|\\|=|&=|\\^=|&&|\\|\\||&|\\||\\+\\+|--|\\%|\\^|\\~|==|!=|::|[.]{3}|->|=>|[+\\-*/&]', Operator), ('[|<>=!()\\[\\]{}.,;:\\?]', Punctuation), ('[^\\W\\d]\\w*', Name.Other)]}

    def analyse_text(text):
        result = 0
        if 'forall' in text:
            result += 0.1
        if 'type' in text:
            result += 0.1
        if 'Self' in text:
            result += 0.1
        if 'observe' in text:
            result += 0.1
        if 'package' in text:
            result += 0.1
        if 'library' in text:
            result += 0.1
        if 'choice' in text:
            result += 0.1
        if 'addr' in text:
            result += 0.1
        if 'constraint' in text:
            result += 0.1
        if 'impl' in text:
            result += 0.1
        return result