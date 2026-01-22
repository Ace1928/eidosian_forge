from pygments.lexer import RegexLexer, bygroups, do_insertions, include, \
from pygments.token import Comment, Error, Keyword, Name, Number, Operator, \
from pygments.util import ClassNotFound, get_bool_opt
class ArturoLexer(RegexLexer):
    """
    For Arturo source code.

    See `Arturo's Github <https://github.com/arturo-lang/arturo>`_
    and `Arturo's Website <https://arturo-lang.io/>`_.

    .. versionadded:: 2.14.0
    """
    name = 'Arturo'
    aliases = ['arturo', 'art']
    filenames = ['*.art']
    url = 'https://arturo-lang.io/'

    def __init__(self, **options):
        self.handle_annotateds = get_bool_opt(options, 'handle_annotateds', True)
        RegexLexer.__init__(self, **options)

    def handle_annotated_strings(self, match):
        """Adds syntax from another languages inside annotated strings

        match args:
            1:open_string,
            2:exclamation_mark,
            3:lang_name,
            4:space_or_newline,
            5:code,
            6:close_string
        """
        from pygments.lexers import get_lexer_by_name
        yield (match.start(1), String.Double, match.group(1))
        yield (match.start(2), String.Interpol, match.group(2))
        yield (match.start(3), String.Interpol, match.group(3))
        yield (match.start(4), Text.Whitespace, match.group(4))
        lexer = None
        if self.handle_annotateds:
            try:
                lexer = get_lexer_by_name(match.group(3).strip())
            except ClassNotFound:
                pass
        code = match.group(5)
        if lexer is None:
            yield (match.group(5), String, code)
        else:
            yield from do_insertions([], lexer.get_tokens_unprocessed(code))
        yield (match.start(6), String.Double, match.group(6))
    tokens = {'root': [(';.*?$', Comment.Single), ('^((\\s#!)|(#!)).*?$', Comment.Hashbang), (words(('false', 'true', 'maybe'), suffix='\\b'), Name.Constant), (words(('this', 'init'), prefix='\\b', suffix='\\b\\??:?'), Name.Builtin.Pseudo), ('`.`', String.Char), ('\\\\\\w+\\b\\??:?', Name.Property), ('#\\w+', Name.Constant), ('\\b[0-9]+\\.[0-9]+', Number.Float), ('\\b[0-9]+', Number.Integer), ('\\w+\\b\\??:', Name.Label), ("\\'(?:\\w+\\b\\??:?)", Keyword.Declaration), ('\\:\\w+', Keyword.Type), ('\\.\\w+\\??:?', Name.Attribute), ('(\\()(.*?)(\\)\\?)', bygroups(Punctuation, using(this), Punctuation)), ('"', String.Double, 'inside-simple-string'), ('»', String.Single, 'inside-smart-string'), ('«««', String.Double, 'inside-safe-string'), ('\\{\\/', String.Single, 'inside-regex-string'), ('\\{\\:', String.Double, 'inside-curly-verb-string'), ('(\\{)(\\!)(\\w+)(\\s|\\n)([\\w\\W]*?)(^\\})', handle_annotated_strings), ('\\{', String.Single, 'inside-curly-string'), ('\\-{3,}', String.Single, 'inside-eof-string'), include('builtin-functions'), ('[()[\\],]', Punctuation), (words(('->', '==>', '|', '::', '@', '#', '$', '&', '!', '!!', './')), Name.Decorator), (words(('<:', ':>', ':<', '>:', '<\\', '<>', '<', '>', 'ø', '∞', '+', '-', '*', '~', '=', '^', '%', '/', '//', '==>', '<=>', '<==>', '=>>', '<<=>>', '<<==>>', '-->', '<->', '<-->', '=|', '|=', '-:', ':-', '_', '.', '..', '\\')), Operator), ('\\b\\w+', Name), ('\\s+', Text.Whitespace), ('.+$', Error)], 'inside-interpol': [('\\|', String.Interpol, '#pop'), ('[^|]+', using(this))], 'inside-template': [('\\|\\|\\>', String.Interpol, '#pop'), ('[^|]+', using(this))], 'string-escape': [(words(('\\\\', '\\n', '\\t', '\\"')), String.Escape)], 'inside-simple-string': [include('string-escape'), ('\\|', String.Interpol, 'inside-interpol'), ('\\<\\|\\|', String.Interpol, 'inside-template'), ('"', String.Double, '#pop'), ('[^|"]+', String)], 'inside-smart-string': [include('string-escape'), ('\\|', String.Interpol, 'inside-interpol'), ('\\<\\|\\|', String.Interpol, 'inside-template'), ('\\n', String.Single, '#pop'), ('[^|\\n]+', String)], 'inside-safe-string': [include('string-escape'), ('\\|', String.Interpol, 'inside-interpol'), ('\\<\\|\\|', String.Interpol, 'inside-template'), ('»»»', String.Double, '#pop'), ('[^|»]+', String)], 'inside-regex-string': [('\\\\[sSwWdDbBZApPxucItnvfr0]+', String.Escape), ('\\|', String.Interpol, 'inside-interpol'), ('\\<\\|\\|', String.Interpol, 'inside-template'), ('\\/\\}', String.Single, '#pop'), ('[^|\\/]+', String.Regex)], 'inside-curly-verb-string': [include('string-escape'), ('\\|', String.Interpol, 'inside-interpol'), ('\\<\\|\\|', String.Interpol, 'inside-template'), ('\\:\\}', String.Double, '#pop'), ('[^|<:]+', String)], 'inside-curly-string': [include('string-escape'), ('\\|', String.Interpol, 'inside-interpol'), ('\\<\\|\\|', String.Interpol, 'inside-template'), ('\\}', String.Single, '#pop'), ('[^|<}]+', String)], 'inside-eof-string': [include('string-escape'), ('\\|', String.Interpol, 'inside-interpol'), ('\\<\\|\\|', String.Interpol, 'inside-template'), ('\\Z', String.Single, '#pop'), ('[^|<]+', String)], 'builtin-functions': [(words(('all', 'and', 'any', 'ascii', 'attr', 'attribute', 'attributeLabel', 'binary', 'blockchar', 'contains', 'database', 'date', 'dictionary', 'empty', 'equal', 'even', 'every', 'exists', 'false', 'floatin', 'function', 'greater', 'greaterOrEqual', 'if', 'in', 'inline', 'integer', 'is', 'key', 'label', 'leap', 'less', 'lessOrEqual', 'literal', 'logical', 'lower', 'nand', 'negative', 'nor', 'not', 'notEqual', 'null', 'numeric', 'odd', 'or', 'path', 'pathLabel', 'positive', 'prefix', 'prime', 'set', 'some', 'sorted', 'standalone', 'string', 'subset', 'suffix', 'superset', 'ymbol', 'true', 'try', 'type', 'unless', 'upper', 'when', 'whitespace', 'word', 'xnor', 'xor', 'zero'), prefix='\\b', suffix='\\b\\?'), Name.Builtin), (words(('abs', 'acos', 'acosh', 'acsec', 'acsech', 'actan', 'actanh', 'add', 'after', 'alphabet', 'and', 'angle', 'append', 'arg', 'args', 'arity', 'array', 'as', 'asec', 'asech', 'asin', 'asinh', 'atan', 'atan2', 'atanh', 'attr', 'attrs', 'average', 'before', 'benchmark', 'blend', 'break', 'builtins1', 'builtins2', 'call', 'capitalize', 'case', 'ceil', 'chop', 'chunk', 'clear', 'close', 'cluster', 'color', 'combine', 'conj', 'continue', 'copy', 'cos', 'cosh', 'couple', 'csec', 'csech', 'ctan', 'ctanh', 'cursor', 'darken', 'dec', 'decode', 'decouple', 'define', 'delete', 'desaturate', 'deviation', 'dictionary', 'difference', 'digest', 'digits', 'div', 'do', 'download', 'drop', 'dup', 'e', 'else', 'empty', 'encode', 'ensure', 'env', 'epsilon', 'escape', 'execute', 'exit', 'exp', 'extend', 'extract', 'factors', 'false', 'fdiv', 'filter', 'first', 'flatten', 'floor', 'fold', 'from', 'function', 'gamma', 'gcd', 'get', 'goto', 'hash', 'help', 'hypot', 'if', 'in', 'inc', 'indent', 'index', 'infinity', 'info', 'input', 'insert', 'inspect', 'intersection', 'invert', 'join', 'keys', 'kurtosis', 'last', 'let', 'levenshtein', 'lighten', 'list', 'ln', 'log', 'loop', 'lower', 'mail', 'map', 'match', 'max', 'maybe', 'median', 'min', 'mod', 'module', 'mul', 'nand', 'neg', 'new', 'nor', 'normalize', 'not', 'now', 'null', 'open', 'or', 'outdent', 'pad', 'panic', 'path', 'pause', 'permissions', 'permutate', 'pi', 'pop', 'pow', 'powerset', 'powmod', 'prefix', 'print', 'prints', 'process', 'product', 'query', 'random', 'range', 'read', 'relative', 'remove', 'rename', 'render', 'repeat', 'replace', 'request', 'return', 'reverse', 'round', 'sample', 'saturate', 'script', 'sec', 'sech', 'select', 'serve', 'set', 'shl', 'shr', 'shuffle', 'sin', 'sinh', 'size', 'skewness', 'slice', 'sort', 'split', 'sqrt', 'squeeze', 'stack', 'strip', 'sub', 'suffix', 'sum', 'switch', 'symbols', 'symlink', 'sys', 'take', 'tan', 'tanh', 'terminal', 'to', 'true', 'truncate', 'try', 'type', 'union', 'unique', 'unless', 'until', 'unzip', 'upper', 'values', 'var', 'variance', 'volume', 'webview', 'while', 'with', 'wordwrap', 'write', 'xnor', 'xor', 'zip'), prefix='\\b', suffix='\\b'), Name.Builtin)]}