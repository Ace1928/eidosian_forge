import re
from pygments.lexer import RegexLexer, include, bygroups, default, using, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import get_bool_opt, iteritems
import pygments.unistring as uni
class LiveScriptLexer(RegexLexer):
    """
    For `LiveScript`_ source code.

    .. _LiveScript: http://gkz.github.com/LiveScript/

    .. versionadded:: 1.6
    """
    name = 'LiveScript'
    aliases = ['live-script', 'livescript']
    filenames = ['*.ls']
    mimetypes = ['text/livescript']
    flags = re.DOTALL
    tokens = {'commentsandwhitespace': [('\\s+', Text), ('/\\*.*?\\*/', Comment.Multiline), ('#.*?\\n', Comment.Single)], 'multilineregex': [include('commentsandwhitespace'), ('//([gim]+\\b|\\B)', String.Regex, '#pop'), ('/', String.Regex), ('[^/#]+', String.Regex)], 'slashstartsregex': [include('commentsandwhitespace'), ('//', String.Regex, ('#pop', 'multilineregex')), ('/(?! )(\\\\.|[^[/\\\\\\n]|\\[(\\\\.|[^\\]\\\\\\n])*])+/([gim]+\\b|\\B)', String.Regex, '#pop'), default('#pop')], 'root': [include('commentsandwhitespace'), ('(?:\\([^()]+\\))?[ ]*[~-]{1,2}>|(?:\\(?[^()\\n]+\\)?)?[ ]*<[~-]{1,2}', Name.Function), ('\\+\\+|&&|(?<![.$])\\b(?:and|x?or|is|isnt|not)\\b|\\?|:|=|\\|\\||\\\\(?=\\n)|(<<|>>>?|==?|!=?|~(?!\\~?>)|-(?!\\-?>)|<(?!\\[)|(?<!\\])>|[+*`%&|^/])=?', Operator, 'slashstartsregex'), ('[{(\\[;,]', Punctuation, 'slashstartsregex'), ('[})\\].]', Punctuation), ('(?<![.$])(for|own|in|of|while|until|loop|break|return|continue|switch|when|then|if|unless|else|throw|try|catch|finally|new|delete|typeof|instanceof|super|extends|this|class|by|const|var|to|til)\\b', Keyword, 'slashstartsregex'), ('(?<![.$])(true|false|yes|no|on|off|null|NaN|Infinity|undefined|void)\\b', Keyword.Constant), ('(Array|Boolean|Date|Error|Function|Math|netscape|Number|Object|Packages|RegExp|String|sun|decodeURI|decodeURIComponent|encodeURI|encodeURIComponent|eval|isFinite|isNaN|parseFloat|parseInt|document|window)\\b', Name.Builtin), ('[$a-zA-Z_][\\w.\\-:$]*\\s*[:=]\\s', Name.Variable, 'slashstartsregex'), ('@[$a-zA-Z_][\\w.\\-:$]*\\s*[:=]\\s', Name.Variable.Instance, 'slashstartsregex'), ('@', Name.Other, 'slashstartsregex'), ('@?[$a-zA-Z_][\\w-]*', Name.Other, 'slashstartsregex'), ('[0-9]+\\.[0-9]+([eE][0-9]+)?[fd]?(?:[a-zA-Z_]+)?', Number.Float), ('[0-9]+(~[0-9a-z]+)?(?:[a-zA-Z_]+)?', Number.Integer), ('"""', String, 'tdqs'), ("'''", String, 'tsqs'), ('"', String, 'dqs'), ("'", String, 'sqs'), ('\\\\\\S+', String), ('<\\[.*?\\]>', String)], 'strings': [('[^#\\\\\\\'"]+', String)], 'interpoling_string': [('\\}', String.Interpol, '#pop'), include('root')], 'dqs': [('"', String, '#pop'), ("\\\\.|\\'", String), ('#\\{', String.Interpol, 'interpoling_string'), ('#', String), include('strings')], 'sqs': [("'", String, '#pop'), ('#|\\\\.|"', String), include('strings')], 'tdqs': [('"""', String, '#pop'), ('\\\\.|\\\'|"', String), ('#\\{', String.Interpol, 'interpoling_string'), ('#', String), include('strings')], 'tsqs': [("'''", String, '#pop'), ('#|\\\\.|\\\'|"', String), include('strings')]}