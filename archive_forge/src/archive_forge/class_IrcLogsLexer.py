import re
from pygments.lexer import RegexLexer, bygroups
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import ClassNotFound
class IrcLogsLexer(RegexLexer):
    """
    Lexer for IRC logs in *irssi*, *xchat* or *weechat* style.
    """
    name = 'IRC logs'
    aliases = ['irc']
    filenames = ['*.weechatlog']
    mimetypes = ['text/x-irclog']
    flags = re.VERBOSE | re.MULTILINE
    timestamp = '\n        (\n          # irssi / xchat and others\n          (?: \\[|\\()?                  # Opening bracket or paren for the timestamp\n            (?:                        # Timestamp\n                (?: (?:\\d{1,4} [-/])*  # Date as - or /-separated groups of digits\n                    (?:\\d{1,4})\n                 [T ])?                # Date/time separator: T or space\n                (?: \\d?\\d [:.])*       # Time as :/.-separated groups of 1 or 2 digits\n                    (?: \\d?\\d)\n            )\n          (?: \\]|\\))?\\s+               # Closing bracket or paren for the timestamp\n        |\n          # weechat\n          \\d{4}\\s\\w{3}\\s\\d{2}\\s        # Date\n          \\d{2}:\\d{2}:\\d{2}\\s+         # Time + Whitespace\n        |\n          # xchat\n          \\w{3}\\s\\d{2}\\s               # Date\n          \\d{2}:\\d{2}:\\d{2}\\s+         # Time + Whitespace\n        )?\n    '
    tokens = {'root': [('^\\*\\*\\*\\*(.*)\\*\\*\\*\\*$', Comment), ('^' + timestamp + '(\\s*<[^>]*>\\s*)$', bygroups(Comment.Preproc, Name.Tag)), ('^' + timestamp + '\n                (\\s*<.*?>\\s*)          # Nick ', bygroups(Comment.Preproc, Name.Tag), 'msg'), ('^' + timestamp + '\n                (\\s*[*]\\s+)            # Star\n                (\\S+\\s+.*?\\n)          # Nick + rest of message ', bygroups(Comment.Preproc, Keyword, Generic.Inserted)), ('^' + timestamp + '\n                (\\s*(?:\\*{3}|<?-[!@=P]?->?)\\s*)  # Star(s) or symbols\n                (\\S+\\s+)                     # Nick + Space\n                (.*?\\n)                         # Rest of message ', bygroups(Comment.Preproc, Keyword, String, Comment)), ('^.*?\\n', Text)], 'msg': [('\\S+:(?!//)', Name.Attribute), ('.*\\n', Text, '#pop')]}