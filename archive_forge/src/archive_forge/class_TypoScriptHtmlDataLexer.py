import re
from pygments.lexer import RegexLexer, include, bygroups, using
from pygments.token import Text, Comment, Name, String, Number, \
class TypoScriptHtmlDataLexer(RegexLexer):
    """
    Lexer that highlights markers, constants and registers within html tags.

    .. versionadded:: 2.2
    """
    name = 'TypoScriptHtmlData'
    aliases = ['typoscripthtmldata']
    tokens = {'root': [('(INCLUDE_TYPOSCRIPT)', Name.Class), ('(EXT|FILE|LLL):[^}\\n"]*', String), ('(.*)(###\\w+###)(.*)', bygroups(String, Name.Constant, String)), ('(\\{)(\\$)((?:[\\w\\-]+\\.)*)([\\w\\-]+)(\\})', bygroups(String.Symbol, Operator, Name.Constant, Name.Constant, String.Symbol)), ('(.*)(\\{)([\\w\\-]+)(\\s*:\\s*)([\\w\\-]+)(\\})(.*)', bygroups(String, String.Symbol, Name.Constant, Operator, Name.Constant, String.Symbol, String)), ('\\s+', Text), ('[<>,:=.*%+|]', String), ('[\\w"\\-!/&;(){}#]+', String)]}