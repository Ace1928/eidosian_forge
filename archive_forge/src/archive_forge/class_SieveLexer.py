from pygments.lexer import RegexLexer, bygroups
from pygments.token import Comment, Name, Literal, String, Text, Punctuation, \
class SieveLexer(RegexLexer):
    """
    Lexer for sieve format.

    .. versionadded:: 2.6
    """
    name = 'Sieve'
    filenames = ['*.siv', '*.sieve']
    aliases = ['sieve']
    tokens = {'root': [('\\s+', Text), ('[();,{}\\[\\]]', Punctuation), ('(?i)require', Keyword.Namespace), ('(?i)(:)(addresses|all|contains|content|create|copy|comparator|count|days|detail|domain|fcc|flags|from|handle|importance|is|localpart|length|lowerfirst|lower|matches|message|mime|options|over|percent|quotewildcard|raw|regex|specialuse|subject|text|under|upperfirst|upper|value)', bygroups(Name.Tag, Name.Tag)), ('(?i)(address|addflag|allof|anyof|body|discard|elsif|else|envelope|ereject|exists|false|fileinto|if|hasflag|header|keep|notify_method_capability|notify|not|redirect|reject|removeflag|setflag|size|spamtest|stop|string|true|vacation|virustest)', Name.Builtin), ('(?i)set', Keyword.Declaration), ('([0-9.]+)([kmgKMG])?', bygroups(Literal.Number, Literal.Number)), ('#.*$', Comment.Single), ('/\\*.*\\*/', Comment.Multiline), ('"[^"]*?"', String), ('text:', Name.Tag, 'text')], 'text': [('[^.].*?\\n', String), ('^\\.', Punctuation, '#pop')]}