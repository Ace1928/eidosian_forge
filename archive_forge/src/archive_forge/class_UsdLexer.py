from pygments.lexer import RegexLexer, bygroups
from pygments.lexer import words as words_
from pygments.lexers._usd_builtins import COMMON_ATTRIBUTES, KEYWORDS, \
from pygments.token import Comment, Keyword, Name, Number, Operator, \
class UsdLexer(RegexLexer):
    """
    A lexer that parses Pixar's Universal Scene Description file format.

    .. versionadded:: 2.6
    """
    name = 'USD'
    url = 'https://graphics.pixar.com/usd/release/index.html'
    aliases = ['usd', 'usda']
    filenames = ['*.usd', '*.usda']
    tokens = {'root': [('(custom){_WHITESPACE}(uniform)(\\s+){}(\\s+){}(\\s*)(=)'.format(_TYPE, _BASE_ATTRIBUTE, _WHITESPACE=_WHITESPACE), bygroups(Keyword.Token, Whitespace, Keyword.Token, Whitespace, Keyword.Type, Whitespace, Name.Attribute, Text, Name.Keyword.Tokens, Whitespace, Operator)), ('(custom){_WHITESPACE}{}(\\s+){}(\\s*)(=)'.format(_TYPE, _BASE_ATTRIBUTE, _WHITESPACE=_WHITESPACE), bygroups(Keyword.Token, Whitespace, Keyword.Type, Whitespace, Name.Attribute, Text, Name.Keyword.Tokens, Whitespace, Operator)), ('(uniform){_WHITESPACE}{}(\\s+){}(\\s*)(=)'.format(_TYPE, _BASE_ATTRIBUTE, _WHITESPACE=_WHITESPACE), bygroups(Keyword.Token, Whitespace, Keyword.Type, Whitespace, Name.Attribute, Text, Name.Keyword.Tokens, Whitespace, Operator)), ('{}{_WHITESPACE}{}(\\s*)(=)'.format(_TYPE, _BASE_ATTRIBUTE, _WHITESPACE=_WHITESPACE), bygroups(Keyword.Type, Whitespace, Name.Attribute, Text, Name.Keyword.Tokens, Whitespace, Operator))] + _keywords(KEYWORDS, Keyword.Tokens) + _keywords(SPECIAL_NAMES, Name.Builtins) + _keywords(COMMON_ATTRIBUTES, Name.Attribute) + [('\\b\\w+:[\\w:]+\\b', Name.Attribute)] + _keywords(OPERATORS, Operator) + [(type_ + '\\[\\]', Keyword.Type) for type_ in TYPES] + _keywords(TYPES, Keyword.Type) + [('[(){}\\[\\]]', Punctuation), ('#.*?$', Comment.Single), (',', Punctuation), (';', Punctuation), ('=', Operator), ('[-]*([0-9]*[.])?[0-9]+(?:e[+-]*\\d+)?', Number), ("'''(?:.|\\n)*?'''", String), ('"""(?:.|\\n)*?"""', String), ("'.*?'", String), ('".*?"', String), ('<(\\.\\./)*([\\w/]+|[\\w/]+\\.\\w+[\\w:]*)>', Name.Namespace), ('@.*?@', String.Interpol), ('\\(.*"[.\\\\n]*".*\\)', String.Doc), ('\\A#usda .+$', Comment.Hashbang), ('\\s+', Whitespace), ('\\w+', Text), ('[_:.]+', Punctuation)]}