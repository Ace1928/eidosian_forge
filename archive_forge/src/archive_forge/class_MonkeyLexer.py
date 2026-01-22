import re
from pygments.lexer import RegexLexer, bygroups, default, words, include
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class MonkeyLexer(RegexLexer):
    """
    For
    `Monkey <https://en.wikipedia.org/wiki/Monkey_(programming_language)>`_
    source code.

    .. versionadded:: 1.6
    """
    name = 'Monkey'
    aliases = ['monkey']
    filenames = ['*.monkey']
    mimetypes = ['text/x-monkey']
    name_variable = '[a-z_]\\w*'
    name_function = '[A-Z]\\w*'
    name_constant = '[A-Z_][A-Z0-9_]*'
    name_class = '[A-Z]\\w*'
    name_module = '[a-z0-9_]*'
    keyword_type = '(?:Int|Float|String|Bool|Object|Array|Void)'
    keyword_type_special = '[?%#$]'
    flags = re.MULTILINE
    tokens = {'root': [('\\s+', Text), ("'.*", Comment), ('(?i)^#rem\\b', Comment.Multiline, 'comment'), ('(?i)^(?:#If|#ElseIf|#Else|#EndIf|#End|#Print|#Error)\\b', Comment.Preproc), ('^#', Comment.Preproc, 'variables'), ('"', String.Double, 'string'), ('[0-9]+\\.[0-9]*(?!\\.)', Number.Float), ('\\.[0-9]+(?!\\.)', Number.Float), ('[0-9]+', Number.Integer), ('\\$[0-9a-fA-Z]+', Number.Hex), ('\\%[10]+', Number.Bin), ('\\b%s\\b' % keyword_type, Keyword.Type), ('(?i)\\b(?:Try|Catch|Throw)\\b', Keyword.Reserved), ('Throwable', Name.Exception), ('(?i)\\b(?:Null|True|False)\\b', Name.Builtin), ('(?i)\\b(?:Self|Super)\\b', Name.Builtin.Pseudo), ('\\b(?:HOST|LANG|TARGET|CONFIG)\\b', Name.Constant), ('(?i)^(Import)(\\s+)(.*)(\\n)', bygroups(Keyword.Namespace, Text, Name.Namespace, Text)), ('(?i)^Strict\\b.*\\n', Keyword.Reserved), ('(?i)(Const|Local|Global|Field)(\\s+)', bygroups(Keyword.Declaration, Text), 'variables'), ('(?i)(New|Class|Interface|Extends|Implements)(\\s+)', bygroups(Keyword.Reserved, Text), 'classname'), ('(?i)(Function|Method)(\\s+)', bygroups(Keyword.Reserved, Text), 'funcname'), ('(?i)(?:End|Return|Public|Private|Extern|Property|Final|Abstract)\\b', Keyword.Reserved), ('(?i)(?:If|Then|Else|ElseIf|EndIf|Select|Case|Default|While|Wend|Repeat|Until|Forever|For|To|Until|Step|EachIn|Next|Exit|Continue)\\s+', Keyword.Reserved), ('(?i)\\b(?:Module|Inline)\\b', Keyword.Reserved), ('[\\[\\]]', Punctuation), ('<=|>=|<>|\\*=|/=|\\+=|-=|&=|~=|\\|=|[-&*/^+=<>|~]', Operator), ('(?i)(?:Not|Mod|Shl|Shr|And|Or)', Operator.Word), ('[(){}!#,.:]', Punctuation), ('%s\\b' % name_constant, Name.Constant), ('%s\\b' % name_function, Name.Function), ('%s\\b' % name_variable, Name.Variable)], 'funcname': [('(?i)%s\\b' % name_function, Name.Function), (':', Punctuation, 'classname'), ('\\s+', Text), ('\\(', Punctuation, 'variables'), ('\\)', Punctuation, '#pop')], 'classname': [('%s\\.' % name_module, Name.Namespace), ('%s\\b' % keyword_type, Keyword.Type), ('%s\\b' % name_class, Name.Class), ('(\\[)(\\s*)(\\d*)(\\s*)(\\])', bygroups(Punctuation, Text, Number.Integer, Text, Punctuation)), ('\\s+(?!<)', Text, '#pop'), ('<', Punctuation, '#push'), ('>', Punctuation, '#pop'), ('\\n', Text, '#pop'), default('#pop')], 'variables': [('%s\\b' % name_constant, Name.Constant), ('%s\\b' % name_variable, Name.Variable), ('%s' % keyword_type_special, Keyword.Type), ('\\s+', Text), (':', Punctuation, 'classname'), (',', Punctuation, '#push'), default('#pop')], 'string': [('[^"~]+', String.Double), ('~q|~n|~r|~t|~z|~~', String.Escape), ('"', String.Double, '#pop')], 'comment': [('(?i)^#rem.*?', Comment.Multiline, '#push'), ('(?i)^#end.*?', Comment.Multiline, '#pop'), ('\\n', Comment.Multiline), ('.+', Comment.Multiline)]}