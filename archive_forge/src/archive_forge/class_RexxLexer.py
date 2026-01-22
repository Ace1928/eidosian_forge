import re
from pygments.lexer import RegexLexer, include, bygroups, default, combined, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import get_bool_opt, get_list_opt, iteritems
class RexxLexer(RegexLexer):
    """
    `Rexx <http://www.rexxinfo.org/>`_ is a scripting language available for
    a wide range of different platforms with its roots found on mainframe
    systems. It is popular for I/O- and data based tasks and can act as glue
    language to bind different applications together.

    .. versionadded:: 2.0
    """
    name = 'Rexx'
    aliases = ['rexx', 'arexx']
    filenames = ['*.rexx', '*.rex', '*.rx', '*.arexx']
    mimetypes = ['text/x-rexx']
    flags = re.IGNORECASE
    tokens = {'root': [('\\s', Whitespace), ('/\\*', Comment.Multiline, 'comment'), ('"', String, 'string_double'), ("'", String, 'string_single'), ('[0-9]+(\\.[0-9]+)?(e[+-]?[0-9])?', Number), ('([a-z_]\\w*)(\\s*)(:)(\\s*)(procedure)\\b', bygroups(Name.Function, Whitespace, Operator, Whitespace, Keyword.Declaration)), ('([a-z_]\\w*)(\\s*)(:)', bygroups(Name.Label, Whitespace, Operator)), include('function'), include('keyword'), include('operator'), ('[a-z_]\\w*', Text)], 'function': [(words(('abbrev', 'abs', 'address', 'arg', 'b2x', 'bitand', 'bitor', 'bitxor', 'c2d', 'c2x', 'center', 'charin', 'charout', 'chars', 'compare', 'condition', 'copies', 'd2c', 'd2x', 'datatype', 'date', 'delstr', 'delword', 'digits', 'errortext', 'form', 'format', 'fuzz', 'insert', 'lastpos', 'left', 'length', 'linein', 'lineout', 'lines', 'max', 'min', 'overlay', 'pos', 'queued', 'random', 'reverse', 'right', 'sign', 'sourceline', 'space', 'stream', 'strip', 'substr', 'subword', 'symbol', 'time', 'trace', 'translate', 'trunc', 'value', 'verify', 'word', 'wordindex', 'wordlength', 'wordpos', 'words', 'x2b', 'x2c', 'x2d', 'xrange'), suffix='(\\s*)(\\()'), bygroups(Name.Builtin, Whitespace, Operator))], 'keyword': [('(address|arg|by|call|do|drop|else|end|exit|for|forever|if|interpret|iterate|leave|nop|numeric|off|on|options|parse|pull|push|queue|return|say|select|signal|to|then|trace|until|while)\\b', Keyword.Reserved)], 'operator': [('(-|//|/|\\(|\\)|\\*\\*|\\*|\\\\<<|\\\\<|\\\\==|\\\\=|\\\\>>|\\\\>|\\\\|\\|\\||\\||&&|&|%|\\+|<<=|<<|<=|<>|<|==|=|><|>=|>>=|>>|>|¬<<|¬<|¬==|¬=|¬>>|¬>|¬|\\.|,)', Operator)], 'string_double': [('[^"\\n]+', String), ('""', String), ('"', String, '#pop'), ('\\n', Text, '#pop')], 'string_single': [("[^\\'\\n]", String), ("\\'\\'", String), ("\\'", String, '#pop'), ('\\n', Text, '#pop')], 'comment': [('[^*]+', Comment.Multiline), ('\\*/', Comment.Multiline, '#pop'), ('\\*', Comment.Multiline)]}
    _c = lambda s: re.compile(s, re.MULTILINE)
    _ADDRESS_COMMAND_PATTERN = _c('^\\s*address\\s+command\\b')
    _ADDRESS_PATTERN = _c('^\\s*address\\s+')
    _DO_WHILE_PATTERN = _c('^\\s*do\\s+while\\b')
    _IF_THEN_DO_PATTERN = _c('^\\s*if\\b.+\\bthen\\s+do\\s*$')
    _PROCEDURE_PATTERN = _c('^\\s*([a-z_]\\w*)(\\s*)(:)(\\s*)(procedure)\\b')
    _ELSE_DO_PATTERN = _c('\\belse\\s+do\\s*$')
    _PARSE_ARG_PATTERN = _c('^\\s*parse\\s+(upper\\s+)?(arg|value)\\b')
    PATTERNS_AND_WEIGHTS = ((_ADDRESS_COMMAND_PATTERN, 0.2), (_ADDRESS_PATTERN, 0.05), (_DO_WHILE_PATTERN, 0.1), (_ELSE_DO_PATTERN, 0.1), (_IF_THEN_DO_PATTERN, 0.1), (_PROCEDURE_PATTERN, 0.5), (_PARSE_ARG_PATTERN, 0.2))

    def analyse_text(text):
        """
        Check for inital comment and patterns that distinguish Rexx from other
        C-like languages.
        """
        if re.search('/\\*\\**\\s*rexx', text, re.IGNORECASE):
            return 1.0
        elif text.startswith('/*'):
            lowerText = text.lower()
            result = sum((weight for pattern, weight in RexxLexer.PATTERNS_AND_WEIGHTS if pattern.search(lowerText))) + 0.01
            return min(result, 1.0)