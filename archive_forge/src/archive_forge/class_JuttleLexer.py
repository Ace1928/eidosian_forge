import re
from pygments.lexer import RegexLexer, include, bygroups, default, using, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import get_bool_opt, iteritems
import pygments.unistring as uni
class JuttleLexer(RegexLexer):
    """
    For `Juttle`_ source code.

    .. _Juttle: https://github.com/juttle/juttle

    """
    name = 'Juttle'
    aliases = ['juttle', 'juttle']
    filenames = ['*.juttle']
    mimetypes = ['application/juttle', 'application/x-juttle', 'text/x-juttle', 'text/juttle']
    flags = re.DOTALL | re.UNICODE | re.MULTILINE
    tokens = {'commentsandwhitespace': [('\\s+', Text), ('//.*?\\n', Comment.Single), ('/\\*.*?\\*/', Comment.Multiline)], 'slashstartsregex': [include('commentsandwhitespace'), ('/(\\\\.|[^[/\\\\\\n]|\\[(\\\\.|[^\\]\\\\\\n])*])+/([gim]+\\b|\\B)', String.Regex, '#pop'), ('(?=/)', Text, ('#pop', 'badregex')), default('#pop')], 'badregex': [('\\n', Text, '#pop')], 'root': [('^(?=\\s|/)', Text, 'slashstartsregex'), include('commentsandwhitespace'), (':\\d{2}:\\d{2}:\\d{2}(\\.\\d*)?:', String.Moment), (':(now|beginning|end|forever|yesterday|today|tomorrow|(\\d+(\\.\\d*)?|\\.\\d+)(ms|[smhdwMy])?):', String.Moment), (':\\d{4}-\\d{2}-\\d{2}(T\\d{2}:\\d{2}:\\d{2}(\\.\\d*)?)?(Z|[+-]\\d{2}:\\d{2}|[+-]\\d{4})?:', String.Moment), (':((\\d+(\\.\\d*)?|\\.\\d+)[ ]+)?(millisecond|second|minute|hour|day|week|month|year)[s]?(([ ]+and[ ]+(\\d+[ ]+)?(millisecond|second|minute|hour|day|week|month|year)[s]?)|[ ]+(ago|from[ ]+now))*:', String.Moment), ('\\+\\+|--|~|&&|\\?|:|\\|\\||\\\\(?=\\n)|(==?|!=?|[-<>+*%&|^/])=?', Operator, 'slashstartsregex'), ('[{(\\[;,]', Punctuation, 'slashstartsregex'), ('[})\\].]', Punctuation), ('(import|return|continue|if|else)\\b', Keyword, 'slashstartsregex'), ('(var|const|function|reducer|sub|input)\\b', Keyword.Declaration, 'slashstartsregex'), ('(batch|emit|filter|head|join|keep|pace|pass|put|read|reduce|remove|sequence|skip|sort|split|tail|unbatch|uniq|view|write)\\b', Keyword.Reserved), ('(true|false|null|Infinity)\\b', Keyword.Constant), ('(Array|Date|Juttle|Math|Number|Object|RegExp|String)\\b', Name.Builtin), (JS_IDENT, Name.Other), ('[0-9][0-9]*\\.[0-9]+([eE][0-9]+)?[fd]?', Number.Float), ('[0-9]+', Number.Integer), ('"(\\\\\\\\|\\\\"|[^"])*"', String.Double), ("'(\\\\\\\\|\\\\'|[^'])*'", String.Single)]}