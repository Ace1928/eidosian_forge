import re
from pygments.lexer import ExtendedRegexLexer, RegexLexer, bygroups, words, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class RslLexer(RegexLexer):
    """
    `RSL <http://en.wikipedia.org/wiki/RAISE>`_ is the formal specification
    language used in RAISE (Rigorous Approach to Industrial Software Engineering)
    method.

    .. versionadded:: 2.0
    """
    name = 'RSL'
    aliases = ['rsl']
    filenames = ['*.rsl']
    mimetypes = ['text/rsl']
    flags = re.MULTILINE | re.DOTALL
    tokens = {'root': [(words(('Bool', 'Char', 'Int', 'Nat', 'Real', 'Text', 'Unit', 'abs', 'all', 'always', 'any', 'as', 'axiom', 'card', 'case', 'channel', 'chaos', 'class', 'devt_relation', 'dom', 'elems', 'else', 'elif', 'end', 'exists', 'extend', 'false', 'for', 'hd', 'hide', 'if', 'in', 'is', 'inds', 'initialise', 'int', 'inter', 'isin', 'len', 'let', 'local', 'ltl_assertion', 'object', 'of', 'out', 'post', 'pre', 'read', 'real', 'rng', 'scheme', 'skip', 'stop', 'swap', 'then', 'theory', 'test_case', 'tl', 'transition_system', 'true', 'type', 'union', 'until', 'use', 'value', 'variable', 'while', 'with', 'write', '~isin', '-inflist', '-infset', '-list', '-set'), prefix='\\b', suffix='\\b'), Keyword), ('(variable|value)\\b', Keyword.Declaration), ('--.*?\\n', Comment), ('<:.*?:>', Comment), ('\\{!.*?!\\}', Comment), ('/\\*.*?\\*/', Comment), ('^[ \\t]*([\\w]+)[ \\t]*:[^:]', Name.Function), ('(^[ \\t]*)([\\w]+)([ \\t]*\\([\\w\\s,]*\\)[ \\t]*)(is|as)', bygroups(Text, Name.Function, Text, Keyword)), ('\\b[A-Z]\\w*\\b', Keyword.Type), ('(true|false)\\b', Keyword.Constant), ('".*"', String), ("\\'.\\'", String.Char), ('(><|->|-m->|/\\\\|<=|<<=|<\\.|\\|\\||\\|\\^\\||-~->|-~m->|\\\\/|>=|>>|\\.>|\\+\\+|-\\\\|<->|=>|:-|~=|\\*\\*|<<|>>=|\\+>|!!|\\|=\\||#)', Operator), ('[0-9]+\\.[0-9]+([eE][0-9]+)?[fd]?', Number.Float), ('0x[0-9a-f]+', Number.Hex), ('[0-9]+', Number.Integer), ('.', Text)]}

    def analyse_text(text):
        """
        Check for the most common text in the beginning of a RSL file.
        """
        if re.search('scheme\\s*.*?=\\s*class\\s*type', text, re.I) is not None:
            return 1.0