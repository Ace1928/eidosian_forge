import re
from pygments.lexer import RegexLexer, ExtendedRegexLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import shebang_matches
class Perl6Lexer(ExtendedRegexLexer):
    """
    For `Perl 6 <http://www.perl6.org>`_ source code.

    .. versionadded:: 2.0
    """
    name = 'Perl6'
    aliases = ['perl6', 'pl6']
    filenames = ['*.pl', '*.pm', '*.nqp', '*.p6', '*.6pl', '*.p6l', '*.pl6', '*.6pm', '*.p6m', '*.pm6', '*.t']
    mimetypes = ['text/x-perl6', 'application/x-perl6']
    flags = re.MULTILINE | re.DOTALL | re.UNICODE
    PERL6_IDENTIFIER_RANGE = "['\\w:-]"
    PERL6_KEYWORDS = ('BEGIN', 'CATCH', 'CHECK', 'CONTROL', 'END', 'ENTER', 'FIRST', 'INIT', 'KEEP', 'LAST', 'LEAVE', 'NEXT', 'POST', 'PRE', 'START', 'TEMP', 'UNDO', 'as', 'assoc', 'async', 'augment', 'binary', 'break', 'but', 'cached', 'category', 'class', 'constant', 'contend', 'continue', 'copy', 'deep', 'default', 'defequiv', 'defer', 'die', 'do', 'else', 'elsif', 'enum', 'equiv', 'exit', 'export', 'fail', 'fatal', 'for', 'gather', 'given', 'goto', 'grammar', 'handles', 'has', 'if', 'inline', 'irs', 'is', 'last', 'leave', 'let', 'lift', 'loop', 'looser', 'macro', 'make', 'maybe', 'method', 'module', 'multi', 'my', 'next', 'of', 'ofs', 'only', 'oo', 'ors', 'our', 'package', 'parsed', 'prec', 'proto', 'readonly', 'redo', 'ref', 'regex', 'reparsed', 'repeat', 'require', 'required', 'return', 'returns', 'role', 'rule', 'rw', 'self', 'slang', 'state', 'sub', 'submethod', 'subset', 'supersede', 'take', 'temp', 'tighter', 'token', 'trusts', 'try', 'unary', 'unless', 'until', 'use', 'warn', 'when', 'where', 'while', 'will')
    PERL6_BUILTINS = ('ACCEPTS', 'HOW', 'REJECTS', 'VAR', 'WHAT', 'WHENCE', 'WHERE', 'WHICH', 'WHO', 'abs', 'acos', 'acosec', 'acosech', 'acosh', 'acotan', 'acotanh', 'all', 'any', 'approx', 'arity', 'asec', 'asech', 'asin', 'asinh', 'assuming', 'atan', 'atan2', 'atanh', 'attr', 'bless', 'body', 'by', 'bytes', 'caller', 'callsame', 'callwith', 'can', 'capitalize', 'cat', 'ceiling', 'chars', 'chmod', 'chomp', 'chop', 'chr', 'chroot', 'circumfix', 'cis', 'classify', 'clone', 'close', 'cmp_ok', 'codes', 'comb', 'connect', 'contains', 'context', 'cos', 'cosec', 'cosech', 'cosh', 'cotan', 'cotanh', 'count', 'defined', 'delete', 'diag', 'dies_ok', 'does', 'e', 'each', 'eager', 'elems', 'end', 'eof', 'eval', 'eval_dies_ok', 'eval_elsewhere', 'eval_lives_ok', 'evalfile', 'exists', 'exp', 'first', 'flip', 'floor', 'flunk', 'flush', 'fmt', 'force_todo', 'fork', 'from', 'getc', 'gethost', 'getlogin', 'getpeername', 'getpw', 'gmtime', 'graphs', 'grep', 'hints', 'hyper', 'im', 'index', 'infix', 'invert', 'is_approx', 'is_deeply', 'isa', 'isa_ok', 'isnt', 'iterator', 'join', 'key', 'keys', 'kill', 'kv', 'lastcall', 'lazy', 'lc', 'lcfirst', 'like', 'lines', 'link', 'lives_ok', 'localtime', 'log', 'log10', 'map', 'max', 'min', 'minmax', 'name', 'new', 'nextsame', 'nextwith', 'nfc', 'nfd', 'nfkc', 'nfkd', 'nok_error', 'nonce', 'none', 'normalize', 'not', 'nothing', 'ok', 'once', 'one', 'open', 'opendir', 'operator', 'ord', 'p5chomp', 'p5chop', 'pack', 'pair', 'pairs', 'pass', 'perl', 'pi', 'pick', 'plan', 'plan_ok', 'polar', 'pop', 'pos', 'postcircumfix', 'postfix', 'pred', 'prefix', 'print', 'printf', 'push', 'quasi', 'quotemeta', 'rand', 're', 'read', 'readdir', 'readline', 'reduce', 'reverse', 'rewind', 'rewinddir', 'rindex', 'roots', 'round', 'roundrobin', 'run', 'runinstead', 'sameaccent', 'samecase', 'say', 'sec', 'sech', 'sech', 'seek', 'shape', 'shift', 'sign', 'signature', 'sin', 'sinh', 'skip', 'skip_rest', 'sleep', 'slurp', 'sort', 'splice', 'split', 'sprintf', 'sqrt', 'srand', 'strand', 'subst', 'substr', 'succ', 'sum', 'symlink', 'tan', 'tanh', 'throws_ok', 'time', 'times', 'to', 'todo', 'trim', 'trim_end', 'trim_start', 'true', 'truncate', 'uc', 'ucfirst', 'undef', 'undefine', 'uniq', 'unlike', 'unlink', 'unpack', 'unpolar', 'unshift', 'unwrap', 'use_ok', 'value', 'values', 'vec', 'version_lt', 'void', 'wait', 'want', 'wrap', 'write', 'zip')
    PERL6_BUILTIN_CLASSES = ('Abstraction', 'Any', 'AnyChar', 'Array', 'Associative', 'Bag', 'Bit', 'Blob', 'Block', 'Bool', 'Buf', 'Byte', 'Callable', 'Capture', 'Char', 'Class', 'Code', 'Codepoint', 'Comparator', 'Complex', 'Decreasing', 'Exception', 'Failure', 'False', 'Grammar', 'Grapheme', 'Hash', 'IO', 'Increasing', 'Int', 'Junction', 'KeyBag', 'KeyExtractor', 'KeyHash', 'KeySet', 'KitchenSink', 'List', 'Macro', 'Mapping', 'Match', 'Matcher', 'Method', 'Module', 'Num', 'Object', 'Ordered', 'Ordering', 'OrderingPair', 'Package', 'Pair', 'Positional', 'Proxy', 'Range', 'Rat', 'Regex', 'Role', 'Routine', 'Scalar', 'Seq', 'Set', 'Signature', 'Str', 'StrLen', 'StrPos', 'Sub', 'Submethod', 'True', 'UInt', 'Undef', 'Version', 'Void', 'Whatever', 'bit', 'bool', 'buf', 'buf1', 'buf16', 'buf2', 'buf32', 'buf4', 'buf64', 'buf8', 'complex', 'int', 'int1', 'int16', 'int2', 'int32', 'int4', 'int64', 'int8', 'num', 'rat', 'rat1', 'rat16', 'rat2', 'rat32', 'rat4', 'rat64', 'rat8', 'uint', 'uint1', 'uint16', 'uint2', 'uint32', 'uint4', 'uint64', 'uint8', 'utf16', 'utf32', 'utf8')
    PERL6_OPERATORS = ('X', 'Z', 'after', 'also', 'and', 'andthen', 'before', 'cmp', 'div', 'eq', 'eqv', 'extra', 'ff', 'fff', 'ge', 'gt', 'le', 'leg', 'lt', 'm', 'mm', 'mod', 'ne', 'or', 'orelse', 'rx', 's', 'tr', 'x', 'xor', 'xx', '++', '--', '**', '!', '+', '-', '~', '?', '|', '||', '+^', '~^', '?^', '^', '*', '/', '%', '%%', '+&', '+<', '+>', '~&', '~<', '~>', '?&', 'gcd', 'lcm', '+', '-', '+|', '+^', '~|', '~^', '?|', '?^', '~', '&', '^', 'but', 'does', '<=>', '..', '..^', '^..', '^..^', '!=', '==', '<', '<=', '>', '>=', '~~', '===', '!eqv', '&&', '||', '^^', '//', 'min', 'max', '??', '!!', 'ff', 'fff', 'so', 'not', '<==', '==>', '<<==', '==>>')
    PERL6_BRACKETS = {u'(': u')', u'<': u'>', u'[': u']', u'{': u'}', u'«': u'»', u'༺': u'༻', u'༼': u'༽', u'᚛': u'᚜', u'‘': u'’', u'‚': u'’', u'‛': u'’', u'“': u'”', u'„': u'”', u'‟': u'”', u'‹': u'›', u'⁅': u'⁆', u'⁽': u'⁾', u'₍': u'₎', u'∈': u'∋', u'∉': u'∌', u'∊': u'∍', u'∕': u'⧵', u'∼': u'∽', u'≃': u'⋍', u'≒': u'≓', u'≔': u'≕', u'≤': u'≥', u'≦': u'≧', u'≨': u'≩', u'≪': u'≫', u'≮': u'≯', u'≰': u'≱', u'≲': u'≳', u'≴': u'≵', u'≶': u'≷', u'≸': u'≹', u'≺': u'≻', u'≼': u'≽', u'≾': u'≿', u'⊀': u'⊁', u'⊂': u'⊃', u'⊄': u'⊅', u'⊆': u'⊇', u'⊈': u'⊉', u'⊊': u'⊋', u'⊏': u'⊐', u'⊑': u'⊒', u'⊘': u'⦸', u'⊢': u'⊣', u'⊦': u'⫞', u'⊨': u'⫤', u'⊩': u'⫣', u'⊫': u'⫥', u'⊰': u'⊱', u'⊲': u'⊳', u'⊴': u'⊵', u'⊶': u'⊷', u'⋉': u'⋊', u'⋋': u'⋌', u'⋐': u'⋑', u'⋖': u'⋗', u'⋘': u'⋙', u'⋚': u'⋛', u'⋜': u'⋝', u'⋞': u'⋟', u'⋠': u'⋡', u'⋢': u'⋣', u'⋤': u'⋥', u'⋦': u'⋧', u'⋨': u'⋩', u'⋪': u'⋫', u'⋬': u'⋭', u'⋰': u'⋱', u'⋲': u'⋺', u'⋳': u'⋻', u'⋴': u'⋼', u'⋶': u'⋽', u'⋷': u'⋾', u'⌈': u'⌉', u'⌊': u'⌋', u'〈': u'〉', u'⎴': u'⎵', u'❨': u'❩', u'❪': u'❫', u'❬': u'❭', u'❮': u'❯', u'❰': u'❱', u'❲': u'❳', u'❴': u'❵', u'⟃': u'⟄', u'⟅': u'⟆', u'⟕': u'⟖', u'⟝': u'⟞', u'⟢': u'⟣', u'⟤': u'⟥', u'⟦': u'⟧', u'⟨': u'⟩', u'⟪': u'⟫', u'⦃': u'⦄', u'⦅': u'⦆', u'⦇': u'⦈', u'⦉': u'⦊', u'⦋': u'⦌', u'⦍': u'⦎', u'⦏': u'⦐', u'⦑': u'⦒', u'⦓': u'⦔', u'⦕': u'⦖', u'⦗': u'⦘', u'⧀': u'⧁', u'⧄': u'⧅', u'⧏': u'⧐', u'⧑': u'⧒', u'⧔': u'⧕', u'⧘': u'⧙', u'⧚': u'⧛', u'⧸': u'⧹', u'⧼': u'⧽', u'⨫': u'⨬', u'⨭': u'⨮', u'⨴': u'⨵', u'⨼': u'⨽', u'⩤': u'⩥', u'⩹': u'⩺', u'⩽': u'⩾', u'⩿': u'⪀', u'⪁': u'⪂', u'⪃': u'⪄', u'⪋': u'⪌', u'⪑': u'⪒', u'⪓': u'⪔', u'⪕': u'⪖', u'⪗': u'⪘', u'⪙': u'⪚', u'⪛': u'⪜', u'⪡': u'⪢', u'⪦': u'⪧', u'⪨': u'⪩', u'⪪': u'⪫', u'⪬': u'⪭', u'⪯': u'⪰', u'⪳': u'⪴', u'⪻': u'⪼', u'⪽': u'⪾', u'⪿': u'⫀', u'⫁': u'⫂', u'⫃': u'⫄', u'⫅': u'⫆', u'⫍': u'⫎', u'⫏': u'⫐', u'⫑': u'⫒', u'⫓': u'⫔', u'⫕': u'⫖', u'⫬': u'⫭', u'⫷': u'⫸', u'⫹': u'⫺', u'⸂': u'⸃', u'⸄': u'⸅', u'⸉': u'⸊', u'⸌': u'⸍', u'⸜': u'⸝', u'⸠': u'⸡', u'〈': u'〉', u'《': u'》', u'「': u'」', u'『': u'』', u'【': u'】', u'〔': u'〕', u'〖': u'〗', u'〘': u'〙', u'〚': u'〛', u'〝': u'〞', u'﴾': u'﴿', u'︗': u'︘', u'︵': u'︶', u'︷': u'︸', u'︹': u'︺', u'︻': u'︼', u'︽': u'︾', u'︿': u'﹀', u'﹁': u'﹂', u'﹃': u'﹄', u'﹇': u'﹈', u'﹙': u'﹚', u'﹛': u'﹜', u'﹝': u'﹞', u'（': u'）', u'＜': u'＞', u'［': u'］', u'｛': u'｝', u'｟': u'｠', u'｢': u'｣'}

    def _build_word_match(words, boundary_regex_fragment=None, prefix='', suffix=''):
        if boundary_regex_fragment is None:
            return '\\b(' + prefix + '|'.join((re.escape(x) for x in words)) + suffix + ')\\b'
        else:
            return '(?<!' + boundary_regex_fragment + ')' + prefix + '(' + '|'.join((re.escape(x) for x in words)) + ')' + suffix + '(?!' + boundary_regex_fragment + ')'

    def brackets_callback(token_class):

        def callback(lexer, match, context):
            groups = match.groupdict()
            opening_chars = groups['delimiter']
            n_chars = len(opening_chars)
            adverbs = groups.get('adverbs')
            closer = Perl6Lexer.PERL6_BRACKETS.get(opening_chars[0])
            text = context.text
            if closer is None:
                end_pos = text.find(opening_chars, match.start('delimiter') + n_chars)
            else:
                closing_chars = closer * n_chars
                nesting_level = 1
                search_pos = match.start('delimiter')
                while nesting_level > 0:
                    next_open_pos = text.find(opening_chars, search_pos + n_chars)
                    next_close_pos = text.find(closing_chars, search_pos + n_chars)
                    if next_close_pos == -1:
                        next_close_pos = len(text)
                        nesting_level = 0
                    elif next_open_pos != -1 and next_open_pos < next_close_pos:
                        nesting_level += 1
                        search_pos = next_open_pos
                    else:
                        nesting_level -= 1
                        search_pos = next_close_pos
                end_pos = next_close_pos
            if end_pos < 0:
                end_pos = len(text)
            if adverbs is not None and re.search(':to\\b', adverbs):
                heredoc_terminator = text[match.start('delimiter') + n_chars:end_pos]
                end_heredoc = re.search('^\\s*' + re.escape(heredoc_terminator) + '\\s*$', text[end_pos:], re.MULTILINE)
                if end_heredoc:
                    end_pos += end_heredoc.end()
                else:
                    end_pos = len(text)
            yield (match.start(), token_class, text[match.start():end_pos + n_chars])
            context.pos = end_pos + n_chars
        return callback

    def opening_brace_callback(lexer, match, context):
        stack = context.stack
        yield (match.start(), Text, context.text[match.start():match.end()])
        context.pos = match.end()
        if len(stack) > 2 and stack[-2] == 'token':
            context.perl6_token_nesting_level += 1

    def closing_brace_callback(lexer, match, context):
        stack = context.stack
        yield (match.start(), Text, context.text[match.start():match.end()])
        context.pos = match.end()
        if len(stack) > 2 and stack[-2] == 'token':
            context.perl6_token_nesting_level -= 1
            if context.perl6_token_nesting_level == 0:
                stack.pop()

    def embedded_perl6_callback(lexer, match, context):
        context.perl6_token_nesting_level = 1
        yield (match.start(), Text, context.text[match.start():match.end()])
        context.pos = match.end()
        context.stack.append('root')
    tokens = {'common': [('#[`|=](?P<delimiter>(?P<first_char>[' + ''.join(PERL6_BRACKETS) + '])(?P=first_char)*)', brackets_callback(Comment.Multiline)), ('#[^\\n]*$', Comment.Singleline), ('^(\\s*)=begin\\s+(\\w+)\\b.*?^\\1=end\\s+\\2', Comment.Multiline), ('^(\\s*)=for.*?\\n\\s*?\\n', Comment.Multiline), ('^=.*?\\n\\s*?\\n', Comment.Multiline), ('(regex|token|rule)(\\s*' + PERL6_IDENTIFIER_RANGE + '+:sym)', bygroups(Keyword, Name), 'token-sym-brackets'), ('(regex|token|rule)(?!' + PERL6_IDENTIFIER_RANGE + ')(\\s*' + PERL6_IDENTIFIER_RANGE + '+)?', bygroups(Keyword, Name), 'pre-token'), ('(role)(\\s+)(q)(\\s*)', bygroups(Keyword, Text, Name, Text)), (_build_word_match(PERL6_KEYWORDS, PERL6_IDENTIFIER_RANGE), Keyword), (_build_word_match(PERL6_BUILTIN_CLASSES, PERL6_IDENTIFIER_RANGE, suffix='(?::[UD])?'), Name.Builtin), (_build_word_match(PERL6_BUILTINS, PERL6_IDENTIFIER_RANGE), Name.Builtin), ('[$@%&][.^:?=!~]?' + PERL6_IDENTIFIER_RANGE + u'+(?:<<.*?>>|<.*?>|«.*?»)*', Name.Variable), ('\\$[!/](?:<<.*?>>|<.*?>|«.*?»)*', Name.Variable.Global), ('::\\?\\w+', Name.Variable.Global), ('[$@%&]\\*' + PERL6_IDENTIFIER_RANGE + u'+(?:<<.*?>>|<.*?>|«.*?»)*', Name.Variable.Global), ('\\$(?:<.*?>)+', Name.Variable), ('(?:q|qq|Q)[a-zA-Z]?\\s*(?P<adverbs>:[\\w\\s:]+)?\\s*(?P<delimiter>(?P<first_char>[^0-9a-zA-Z:\\s])(?P=first_char)*)', brackets_callback(String)), ('0_?[0-7]+(_[0-7]+)*', Number.Oct), ('0x[0-9A-Fa-f]+(_[0-9A-Fa-f]+)*', Number.Hex), ('0b[01]+(_[01]+)*', Number.Bin), ('(?i)(\\d*(_\\d*)*\\.\\d+(_\\d*)*|\\d+(_\\d*)*\\.\\d+(_\\d*)*)(e[+-]?\\d+)?', Number.Float), ('(?i)\\d+(_\\d*)*e[+-]?\\d+(_\\d*)*', Number.Float), ('\\d+(_\\d+)*', Number.Integer), ('(?<=~~)\\s*/(?:\\\\\\\\|\\\\/|.)*?/', String.Regex), ('(?<=[=(,])\\s*/(?:\\\\\\\\|\\\\/|.)*?/', String.Regex), ('m\\w+(?=\\()', Name), ('(?:m|ms|rx)\\s*(?P<adverbs>:[\\w\\s:]+)?\\s*(?P<delimiter>(?P<first_char>[^\\w:\\s])(?P=first_char)*)', brackets_callback(String.Regex)), ('(?:s|ss|tr)\\s*(?::[\\w\\s:]+)?\\s*/(?:\\\\\\\\|\\\\/|.)*?/(?:\\\\\\\\|\\\\/|.)*?/', String.Regex), ('<[^\\s=].*?\\S>', String), (_build_word_match(PERL6_OPERATORS), Operator), ('\\w' + PERL6_IDENTIFIER_RANGE + '*', Name), ("'(\\\\\\\\|\\\\[^\\\\]|[^'\\\\])*'", String), ('"(\\\\\\\\|\\\\[^\\\\]|[^"\\\\])*"', String)], 'root': [include('common'), ('\\{', opening_brace_callback), ('\\}', closing_brace_callback), ('.+?', Text)], 'pre-token': [include('common'), ('\\{', Text, ('#pop', 'token')), ('.+?', Text)], 'token-sym-brackets': [('(?P<delimiter>(?P<first_char>[' + ''.join(PERL6_BRACKETS) + '])(?P=first_char)*)', brackets_callback(Name), ('#pop', 'pre-token')), default(('#pop', 'pre-token'))], 'token': [('\\}', Text, '#pop'), ('(?<=:)(?:my|our|state|constant|temp|let).*?;', using(this)), ('<(?:[-!?+.]\\s*)?\\[.*?\\]>', String.Regex), ("(?<!\\\\)'(\\\\\\\\|\\\\[^\\\\]|[^'\\\\])*'", String.Regex), ('(?<!\\\\)"(\\\\\\\\|\\\\[^\\\\]|[^"\\\\])*"', String.Regex), ('#.*?$', Comment.Singleline), ('\\{', embedded_perl6_callback), ('.+?', String.Regex)]}

    def analyse_text(text):

        def strip_pod(lines):
            in_pod = False
            stripped_lines = []
            for line in lines:
                if re.match('^=(?:end|cut)', line):
                    in_pod = False
                elif re.match('^=\\w+', line):
                    in_pod = True
                elif not in_pod:
                    stripped_lines.append(line)
            return stripped_lines
        lines = text.splitlines()
        lines = strip_pod(lines)
        text = '\n'.join(lines)
        if shebang_matches(text, 'perl6|rakudo|niecza|pugs'):
            return True
        saw_perl_decl = False
        rating = False
        if re.search('(?:my|our|has)\\s+(?:' + Perl6Lexer.PERL6_IDENTIFIER_RANGE + '+\\s+)?[$@%&(]', text):
            rating = 0.8
            saw_perl_decl = True
        for line in lines:
            line = re.sub('#.*', '', line)
            if re.match('^\\s*$', line):
                continue
            if re.match('^\\s*(?:use\\s+)?v6(?:\\.\\d(?:\\.\\d)?)?;', line):
                return True
            class_decl = re.match('^\\s*(?:(?P<scope>my|our)\\s+)?(?:module|class|role|enum|grammar)', line)
            if class_decl:
                if saw_perl_decl or class_decl.group('scope') is not None:
                    return True
                rating = 0.05
                continue
            break
        return rating

    def __init__(self, **options):
        super(Perl6Lexer, self).__init__(**options)
        self.encoding = options.get('encoding', 'utf-8')