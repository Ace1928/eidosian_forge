import re
from pygments.lexer import RegexLexer, include, bygroups, using
from pygments.token import Text, Comment, Name, String, Number, \
class TypoScriptCssDataLexer(RegexLexer):
    """
    Lexer that highlights markers, constants and registers within css blocks.

    .. versionadded:: 2.2
    """
    name = 'TypoScriptCssData'
    aliases = ['typoscriptcssdata']
    tokens = {'root': [('(.*)(###\\w+###)(.*)', bygroups(String, Name.Constant, String)), ('(\\{)(\\$)((?:[\\w\\-]+\\.)*)([\\w\\-]+)(\\})', bygroups(String.Symbol, Operator, Name.Constant, Name.Constant, String.Symbol)), ('(.*)(\\{)([\\w\\-]+)(\\s*:\\s*)([\\w\\-]+)(\\})(.*)', bygroups(String, String.Symbol, Name.Constant, Operator, Name.Constant, String.Symbol, String)), ('\\s+', Text), ('/\\*(?:(?!\\*/).)*\\*/', Comment), ('(?<!(#|\\\'|"))(?:#(?!(?:[a-fA-F0-9]{6}|[a-fA-F0-9]{3}))[^\\n#]+|//[^\\n]*)', Comment), ('[<>,:=.*%+|]', String), ('[\\w"\\-!/&;(){}]+', String)]}