from pygments.lexer import RegexLexer, include, bygroups
from pygments.token import Comment, Keyword, Name, String, Number, Generic, Text
class TAPLexer(RegexLexer):
    """
    For Test Anything Protocol (TAP) output.

    .. versionadded:: 2.1
    """
    name = 'TAP'
    aliases = ['tap']
    filenames = ['*.tap']
    tokens = {'root': [('^TAP version \\d+\\n', Name.Namespace), ('^1\\.\\.\\d+', Keyword.Declaration, 'plan'), ('^(not ok)([^\\S\\n]*)(\\d*)', bygroups(Generic.Error, Text, Number.Integer), 'test'), ('^(ok)([^\\S\\n]*)(\\d*)', bygroups(Keyword.Reserved, Text, Number.Integer), 'test'), ('^#.*\\n', Comment), ('^Bail out!.*\\n', Generic.Error), ('^.*\\n', Text)], 'plan': [('[^\\S\\n]+', Text), ('#', Comment, 'directive'), ('\\n', Comment, '#pop'), ('.*\\n', Generic.Error, '#pop')], 'test': [('[^\\S\\n]+', Text), ('#', Comment, 'directive'), ('\\S+', Text), ('\\n', Text, '#pop')], 'directive': [('[^\\S\\n]+', Comment), ('(?i)\\bTODO\\b', Comment.Preproc), ('(?i)\\bSKIP\\S*', Comment.Preproc), ('\\S+', Comment), ('\\n', Comment, '#pop:2')]}