from pygments.lexer import RegexLexer, DelegatingLexer, bygroups
from pygments.lexers.mime import MIMELexer
from pygments.token import Text, Keyword, Name, String, Number, Comment
from pygments.util import get_bool_opt
class EmailHeaderLexer(RegexLexer):
    """
    Sub-lexer for raw E-mail. This lexer only process header part of e-mail.

    .. versionadded:: 2.5
    """

    def __init__(self, **options):
        super().__init__(**options)
        self.highlight_x = get_bool_opt(options, 'highlight-X-header', False)

    def get_x_header_tokens(self, match):
        if self.highlight_x:
            yield (match.start(1), Name.Tag, match.group(1))
            default_actions = self.get_tokens_unprocessed(match.group(2), stack=('root', 'header'))
            yield from default_actions
        else:
            yield (match.start(1), Comment.Special, match.group(1))
            yield (match.start(2), Comment.Multiline, match.group(2))
    tokens = {'root': [('^(?:[A-WYZ]|X400)[\\w\\-]*:', Name.Tag, 'header'), ('^(X-(?:\\w[\\w\\-]*:))([\\s\\S]*?\\n)(?![ \\t])', get_x_header_tokens)], 'header': [('\\n[ \\t]', Text.Whitespace), ('\\n(?![ \\t])', Text.Whitespace, '#pop'), ('\\bE?SMTPS?\\b', Keyword), ('\\b(?:HE|EH)LO\\b', Keyword), ('[\\w\\.\\-\\+=]+@[\\w\\.\\-]+', Name.Label), ('<[\\w\\.\\-\\+=]+@[\\w\\.\\-]+>', Name.Label), ('\\b(\\w[\\w\\.-]*\\.[\\w\\.-]*\\w[a-zA-Z]+)\\b', Name.Function), ('(?<=\\b)(?:(?:25[0-5]|2[0-4][0-9]|1?[0-9][0-9]?)\\.){3}(?:25[0-5]|2[0-4][0-9]|1?[0-9][0-9]?)(?=\\b)', Number.Integer), ('(?<=\\b)([0-9a-fA-F]{1,4}:){1,7}:(?!\\b)', Number.Hex), ('(?<=\\b):((:[0-9a-fA-F]{1,4}){1,7}|:)(?=\\b)', Number.Hex), ('(?<=\\b)([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}(?=\\b)', Number.Hex), ('(?<=\\b)([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}(?=\\b)', Number.Hex), ('(?<=\\b)[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})(?=\\b)', Number.Hex), ('(?<=\\b)fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}(?=\\b)', Number.Hex), ('(?<=\\b)([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}(?=\\b)', Number.Hex), ('(?<=\\b)([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}(?=\\b)', Number.Hex), ('(?<=\\b)([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}(?=\\b)', Number.Hex), ('(?<=\\b)([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}(?=\\b)', Number.Hex), ('(?<=\\b)::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])(?=\\b)', Number.Hex), ('(?<=\\b)([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])(?=\\b)', Number.Hex), ('(?:(Sun|Mon|Tue|Wed|Thu|Fri|Sat),\\s+)?(0[1-9]|[1-2]?[0-9]|3[01])\\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\\s+(19[0-9]{2}|[2-9][0-9]{3})\\s+(2[0-3]|[0-1][0-9]):([0-5][0-9])(?::(60|[0-5][0-9]))?(?:\\.\\d{1,5})?\\s+([-\\+][0-9]{2}[0-5][0-9]|\\(?(?:UTC?|GMT|(?:E|C|M|P)(?:ST|ET|DT)|[A-IK-Z])\\)?)', Name.Decorator), ('(=\\?)([\\w-]+)(\\?)([BbQq])(\\?)([\\[\\w!\\"#$%&\\\'()*+,-./:;<=>@[\\\\\\]^_`{|}~]+)(\\?=)', bygroups(String.Affix, Name.Constant, String.Affix, Keyword.Constant, String.Affix, Number.Hex, String.Affix)), ('[\\s]+', Text.Whitespace), ('[\\S]', Text)]}