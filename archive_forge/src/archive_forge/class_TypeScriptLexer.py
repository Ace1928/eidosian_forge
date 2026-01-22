import re
from pygments.lexer import RegexLexer, include, bygroups, default, using, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import get_bool_opt, iteritems
import pygments.unistring as uni
class TypeScriptLexer(RegexLexer):
    """
    For `TypeScript <http://typescriptlang.org/>`_ source code.

    .. versionadded:: 1.6
    """
    name = 'TypeScript'
    aliases = ['ts', 'typescript']
    filenames = ['*.ts', '*.tsx']
    mimetypes = ['text/x-typescript']
    flags = re.DOTALL | re.MULTILINE
    tokens = {'commentsandwhitespace': [('\\s+', Text), ('<!--', Comment), ('//.*?\\n', Comment.Single), ('/\\*.*?\\*/', Comment.Multiline)], 'slashstartsregex': [include('commentsandwhitespace'), ('/(\\\\.|[^[/\\\\\\n]|\\[(\\\\.|[^\\]\\\\\\n])*])+/([gim]+\\b|\\B)', String.Regex, '#pop'), ('(?=/)', Text, ('#pop', 'badregex')), default('#pop')], 'badregex': [('\\n', Text, '#pop')], 'root': [('^(?=\\s|/|<!--)', Text, 'slashstartsregex'), include('commentsandwhitespace'), ('\\+\\+|--|~|&&|\\?|:|\\|\\||\\\\(?=\\n)|(<<|>>>?|==?|!=?|[-<>+*%&|^/])=?', Operator, 'slashstartsregex'), ('[{(\\[;,]', Punctuation, 'slashstartsregex'), ('[})\\].]', Punctuation), ('(for|in|while|do|break|return|continue|switch|case|default|if|else|throw|try|catch|finally|new|delete|typeof|instanceof|void|this)\\b', Keyword, 'slashstartsregex'), ('(var|let|with|function)\\b', Keyword.Declaration, 'slashstartsregex'), ('(abstract|boolean|byte|char|class|const|debugger|double|enum|export|extends|final|float|goto|implements|import|int|interface|long|native|package|private|protected|public|short|static|super|synchronized|throws|transient|volatile)\\b', Keyword.Reserved), ('(true|false|null|NaN|Infinity|undefined)\\b', Keyword.Constant), ('(Array|Boolean|Date|Error|Function|Math|netscape|Number|Object|Packages|RegExp|String|sun|decodeURI|decodeURIComponent|encodeURI|encodeURIComponent|Error|eval|isFinite|isNaN|parseFloat|parseInt|document|this|window)\\b', Name.Builtin), ('\\b(module)(\\s*)(\\s*[\\w?.$][\\w?.$]*)(\\s*)', bygroups(Keyword.Reserved, Text, Name.Other, Text), 'slashstartsregex'), ('\\b(string|bool|number)\\b', Keyword.Type), ('\\b(constructor|declare|interface|as|AS)\\b', Keyword.Reserved), ('(super)(\\s*)(\\([\\w,?.$\\s]+\\s*\\))', bygroups(Keyword.Reserved, Text), 'slashstartsregex'), ('([a-zA-Z_?.$][\\w?.$]*)\\(\\) \\{', Name.Other, 'slashstartsregex'), ('([\\w?.$][\\w?.$]*)(\\s*:\\s*)([\\w?.$][\\w?.$]*)', bygroups(Name.Other, Text, Keyword.Type)), ('[$a-zA-Z_]\\w*', Name.Other), ('[0-9][0-9]*\\.[0-9]+([eE][0-9]+)?[fd]?', Number.Float), ('0x[0-9a-fA-F]+', Number.Hex), ('[0-9]+', Number.Integer), ('"(\\\\\\\\|\\\\"|[^"])*"', String.Double), ("'(\\\\\\\\|\\\\'|[^'])*'", String.Single), ('`', String.Backtick, 'interp'), ('@\\w+', Keyword.Declaration)], 'interp': [('`', String.Backtick, '#pop'), ('\\\\\\\\', String.Backtick), ('\\\\`', String.Backtick), ('\\$\\{', String.Interpol, 'interp-inside'), ('\\$', String.Backtick), ('[^`\\\\$]+', String.Backtick)], 'interp-inside': [('\\}', String.Interpol, '#pop'), include('root')]}

    def analyse_text(text):
        if re.search('^(import.+(from\\s+)?["\']|(export\\s*)?(interface|class|function)\\s+)', text, re.MULTILINE):
            return 1.0