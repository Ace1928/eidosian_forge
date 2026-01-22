import re
from pygments.lexer import RegexLexer, include, bygroups, default, using, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import get_bool_opt, iteritems
import pygments.unistring as uni
class CoffeeScriptLexer(RegexLexer):
    """
    For `CoffeeScript`_ source code.

    .. _CoffeeScript: http://coffeescript.org

    .. versionadded:: 1.3
    """
    name = 'CoffeeScript'
    aliases = ['coffee-script', 'coffeescript', 'coffee']
    filenames = ['*.coffee']
    mimetypes = ['text/coffeescript']
    _operator_re = '\\+\\+|~|&&|\\band\\b|\\bor\\b|\\bis\\b|\\bisnt\\b|\\bnot\\b|\\?|:|\\|\\||\\\\(?=\\n)|(<<|>>>?|==?(?!>)|!=?|=(?!>)|-(?!>)|[<>+*`%&\\|\\^/])=?'
    flags = re.DOTALL
    tokens = {'commentsandwhitespace': [('\\s+', Text), ('###[^#].*?###', Comment.Multiline), ('#(?!##[^#]).*?\\n', Comment.Single)], 'multilineregex': [('[^/#]+', String.Regex), ('///([gim]+\\b|\\B)', String.Regex, '#pop'), ('#\\{', String.Interpol, 'interpoling_string'), ('[/#]', String.Regex)], 'slashstartsregex': [include('commentsandwhitespace'), ('///', String.Regex, ('#pop', 'multilineregex')), ('/(?! )(\\\\.|[^[/\\\\\\n]|\\[(\\\\.|[^\\]\\\\\\n])*])+/([gim]+\\b|\\B)', String.Regex, '#pop'), ('/', Operator), default('#pop')], 'root': [include('commentsandwhitespace'), ('^(?=\\s|/)', Text, 'slashstartsregex'), (_operator_re, Operator, 'slashstartsregex'), ('(?:\\([^()]*\\))?\\s*[=-]>', Name.Function, 'slashstartsregex'), ('[{(\\[;,]', Punctuation, 'slashstartsregex'), ('[})\\].]', Punctuation), ('(?<![.$])(for|own|in|of|while|until|loop|break|return|continue|switch|when|then|if|unless|else|throw|try|catch|finally|new|delete|typeof|instanceof|super|extends|this|class|by)\\b', Keyword, 'slashstartsregex'), ('(?<![.$])(true|false|yes|no|on|off|null|NaN|Infinity|undefined)\\b', Keyword.Constant), ('(Array|Boolean|Date|Error|Function|Math|netscape|Number|Object|Packages|RegExp|String|sun|decodeURI|decodeURIComponent|encodeURI|encodeURIComponent|eval|isFinite|isNaN|parseFloat|parseInt|document|window)\\b', Name.Builtin), ('[$a-zA-Z_][\\w.:$]*\\s*[:=]\\s', Name.Variable, 'slashstartsregex'), ('@[$a-zA-Z_][\\w.:$]*\\s*[:=]\\s', Name.Variable.Instance, 'slashstartsregex'), ('@', Name.Other, 'slashstartsregex'), ('@?[$a-zA-Z_][\\w$]*', Name.Other), ('[0-9][0-9]*\\.[0-9]+([eE][0-9]+)?[fd]?', Number.Float), ('0x[0-9a-fA-F]+', Number.Hex), ('[0-9]+', Number.Integer), ('"""', String, 'tdqs'), ("'''", String, 'tsqs'), ('"', String, 'dqs'), ("'", String, 'sqs')], 'strings': [('[^#\\\\\\\'"]+', String)], 'interpoling_string': [('\\}', String.Interpol, '#pop'), include('root')], 'dqs': [('"', String, '#pop'), ("\\\\.|\\'", String), ('#\\{', String.Interpol, 'interpoling_string'), ('#', String), include('strings')], 'sqs': [("'", String, '#pop'), ('#|\\\\.|"', String), include('strings')], 'tdqs': [('"""', String, '#pop'), ('\\\\.|\\\'|"', String), ('#\\{', String.Interpol, 'interpoling_string'), ('#', String), include('strings')], 'tsqs': [("'''", String, '#pop'), ('#|\\\\.|\\\'|"', String), include('strings')]}