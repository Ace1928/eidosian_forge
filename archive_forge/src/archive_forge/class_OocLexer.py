from pygments.lexer import RegexLexer, bygroups, words
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class OocLexer(RegexLexer):
    """
    For `Ooc <http://ooc-lang.org/>`_ source code

    .. versionadded:: 1.2
    """
    name = 'Ooc'
    aliases = ['ooc']
    filenames = ['*.ooc']
    mimetypes = ['text/x-ooc']
    tokens = {'root': [(words(('class', 'interface', 'implement', 'abstract', 'extends', 'from', 'this', 'super', 'new', 'const', 'final', 'static', 'import', 'use', 'extern', 'inline', 'proto', 'break', 'continue', 'fallthrough', 'operator', 'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'as', 'in', 'version', 'return', 'true', 'false', 'null'), prefix='\\b', suffix='\\b'), Keyword), ('include\\b', Keyword, 'include'), ('(cover)([ \\t]+)(from)([ \\t]+)(\\w+[*@]?)', bygroups(Keyword, Text, Keyword, Text, Name.Class)), ('(func)((?:[ \\t]|\\\\\\n)+)(~[a-z_]\\w*)', bygroups(Keyword, Text, Name.Function)), ('\\bfunc\\b', Keyword), ('//.*', Comment), ('(?s)/\\*.*?\\*/', Comment.Multiline), ('(==?|\\+=?|-[=>]?|\\*=?|/=?|:=|!=?|%=?|\\?|>{1,3}=?|<{1,3}=?|\\.\\.|&&?|\\|\\|?|\\^=?)', Operator), ('(\\.)([ \\t]*)([a-z]\\w*)', bygroups(Operator, Text, Name.Function)), ('[A-Z][A-Z0-9_]+', Name.Constant), ('[A-Z]\\w*([@*]|\\[[ \\t]*\\])?', Name.Class), ('([a-z]\\w*(?:~[a-z]\\w*)?)((?:[ \\t]|\\\\\\n)*)(?=\\()', bygroups(Name.Function, Text)), ('[a-z]\\w*', Name.Variable), ('[:(){}\\[\\];,]', Punctuation), ('0x[0-9a-fA-F]+', Number.Hex), ('0c[0-9]+', Number.Oct), ('0b[01]+', Number.Bin), ('[0-9_]\\.[0-9_]*(?!\\.)', Number.Float), ('[0-9_]+', Number.Decimal), ('"(?:\\\\.|\\\\[0-7]{1,3}|\\\\x[a-fA-F0-9]{1,2}|[^\\\\"])*"', String.Double), ("'(?:\\\\.|\\\\[0-9]{1,3}|\\\\x[a-fA-F0-9]{1,2}|[^\\\\\\'\\n])'", String.Char), ('@', Punctuation), ('\\.', Punctuation), ('\\\\[ \\t\\n]', Text), ('[ \\t]+', Text)], 'include': [('[\\w/]+', Name), (',', Punctuation), ('[ \\t]', Text), ('[;\\n]', Text, '#pop')]}