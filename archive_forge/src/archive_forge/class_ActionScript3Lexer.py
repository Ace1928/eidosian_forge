import re
from pygments.lexer import RegexLexer, bygroups, using, this, words, default
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class ActionScript3Lexer(RegexLexer):
    """
    For ActionScript 3 source code.

    .. versionadded:: 0.11
    """
    name = 'ActionScript 3'
    aliases = ['as3', 'actionscript3']
    filenames = ['*.as']
    mimetypes = ['application/x-actionscript3', 'text/x-actionscript3', 'text/actionscript3']
    identifier = '[$a-zA-Z_]\\w*'
    typeidentifier = identifier + '(?:\\.<\\w+>)?'
    flags = re.DOTALL | re.MULTILINE
    tokens = {'root': [('\\s+', Text), ('(function\\s+)(' + identifier + ')(\\s*)(\\()', bygroups(Keyword.Declaration, Name.Function, Text, Operator), 'funcparams'), ('(var|const)(\\s+)(' + identifier + ')(\\s*)(:)(\\s*)(' + typeidentifier + ')', bygroups(Keyword.Declaration, Text, Name, Text, Punctuation, Text, Keyword.Type)), ('(import|package)(\\s+)((?:' + identifier + '|\\.)+)(\\s*)', bygroups(Keyword, Text, Name.Namespace, Text)), ('(new)(\\s+)(' + typeidentifier + ')(\\s*)(\\()', bygroups(Keyword, Text, Keyword.Type, Text, Operator)), ('//.*?\\n', Comment.Single), ('/\\*.*?\\*/', Comment.Multiline), ('/(\\\\\\\\|\\\\/|[^\\n])*/[gisx]*', String.Regex), ('(\\.)(' + identifier + ')', bygroups(Operator, Name.Attribute)), ('(case|default|for|each|in|while|do|break|return|continue|if|else|throw|try|catch|with|new|typeof|arguments|instanceof|this|switch|import|include|as|is)\\b', Keyword), ('(class|public|final|internal|native|override|private|protected|static|import|extends|implements|interface|intrinsic|return|super|dynamic|function|const|get|namespace|package|set)\\b', Keyword.Declaration), ('(true|false|null|NaN|Infinity|-Infinity|undefined|void)\\b', Keyword.Constant), ('(decodeURI|decodeURIComponent|encodeURI|escape|eval|isFinite|isNaN|isXMLName|clearInterval|fscommand|getTimer|getURL|getVersion|isFinite|parseFloat|parseInt|setInterval|trace|updateAfterEvent|unescape)\\b', Name.Function), (identifier, Name), ('[0-9][0-9]*\\.[0-9]+([eE][0-9]+)?[fd]?', Number.Float), ('0x[0-9a-f]+', Number.Hex), ('[0-9]+', Number.Integer), ('"(\\\\\\\\|\\\\"|[^"])*"', String.Double), ("'(\\\\\\\\|\\\\'|[^'])*'", String.Single), ('[~^*!%&<>|+=:;,/?\\\\{}\\[\\]().-]+', Operator)], 'funcparams': [('\\s+', Text), ('(\\s*)(\\.\\.\\.)?(' + identifier + ')(\\s*)(:)(\\s*)(' + typeidentifier + '|\\*)(\\s*)', bygroups(Text, Punctuation, Name, Text, Operator, Text, Keyword.Type, Text), 'defval'), ('\\)', Operator, 'type')], 'type': [('(\\s*)(:)(\\s*)(' + typeidentifier + '|\\*)', bygroups(Text, Operator, Text, Keyword.Type), '#pop:2'), ('\\s+', Text, '#pop:2'), default('#pop:2')], 'defval': [('(=)(\\s*)([^(),]+)(\\s*)(,?)', bygroups(Operator, Text, using(this), Text, Operator), '#pop'), (',', Operator, '#pop'), default('#pop')]}

    def analyse_text(text):
        if re.match('\\w+\\s*:\\s*\\w', text):
            return 0.3
        return 0