import re
from pygments.lexer import ExtendedRegexLexer, include, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class CrystalLexer(ExtendedRegexLexer):
    """
    For `Crystal <http://crystal-lang.org>`_ source code.

    .. versionadded:: 2.2
    """
    name = 'Crystal'
    aliases = ['cr', 'crystal']
    filenames = ['*.cr']
    mimetypes = ['text/x-crystal']
    flags = re.DOTALL | re.MULTILINE

    def heredoc_callback(self, match, ctx):
        start = match.start(1)
        yield (start, Operator, match.group(1))
        yield (match.start(2), String.Heredoc, match.group(2))
        yield (match.start(3), String.Delimiter, match.group(3))
        yield (match.start(4), String.Heredoc, match.group(4))
        heredocstack = ctx.__dict__.setdefault('heredocstack', [])
        outermost = not bool(heredocstack)
        heredocstack.append((match.group(1) == '<<-', match.group(3)))
        ctx.pos = match.start(5)
        ctx.end = match.end(5)
        for i, t, v in self.get_tokens_unprocessed(context=ctx):
            yield (i, t, v)
        ctx.pos = match.end()
        if outermost:
            for tolerant, hdname in heredocstack:
                lines = []
                for match in line_re.finditer(ctx.text, ctx.pos):
                    if tolerant:
                        check = match.group().strip()
                    else:
                        check = match.group().rstrip()
                    if check == hdname:
                        for amatch in lines:
                            yield (amatch.start(), String.Heredoc, amatch.group())
                        yield (match.start(), String.Delimiter, match.group())
                        ctx.pos = match.end()
                        break
                    else:
                        lines.append(match)
                else:
                    for amatch in lines:
                        yield (amatch.start(), Error, amatch.group())
            ctx.end = len(ctx.text)
            del heredocstack[:]

    def gen_crystalstrings_rules():

        def intp_regex_callback(self, match, ctx):
            yield (match.start(1), String.Regex, match.group(1))
            nctx = LexerContext(match.group(3), 0, ['interpolated-regex'])
            for i, t, v in self.get_tokens_unprocessed(context=nctx):
                yield (match.start(3) + i, t, v)
            yield (match.start(4), String.Regex, match.group(4))
            ctx.pos = match.end()

        def intp_string_callback(self, match, ctx):
            yield (match.start(1), String.Other, match.group(1))
            nctx = LexerContext(match.group(3), 0, ['interpolated-string'])
            for i, t, v in self.get_tokens_unprocessed(context=nctx):
                yield (match.start(3) + i, t, v)
            yield (match.start(4), String.Other, match.group(4))
            ctx.pos = match.end()
        states = {}
        states['strings'] = [('\\:@{0,2}[a-zA-Z_]\\w*[!?]?', String.Symbol), (words(CRYSTAL_OPERATORS, prefix='\\:@{0,2}'), String.Symbol), (":'(\\\\\\\\|\\\\'|[^'])*'", String.Symbol), ("'(\\\\\\\\|\\\\'|[^']|\\\\[^'\\\\]+)'", String.Char), (':"', String.Symbol, 'simple-sym'), ('([a-zA-Z_]\\w*)(:)(?!:)', bygroups(String.Symbol, Punctuation)), ('"', String.Double, 'simple-string'), ('(?<!\\.)`', String.Backtick, 'simple-backtick')]
        for name, ttype, end in (('string', String.Double, '"'), ('sym', String.Symbol, '"'), ('backtick', String.Backtick, '`')):
            states['simple-' + name] = [include('string-escaped' if name == 'sym' else 'string-intp-escaped'), ('[^\\\\%s#]+' % end, ttype), ('[\\\\#]', ttype), (end, ttype, '#pop')]
        for lbrace, rbrace, bracecc, name in (('\\{', '\\}', '{}', 'cb'), ('\\[', '\\]', '\\[\\]', 'sb'), ('\\(', '\\)', '()', 'pa'), ('<', '>', '<>', 'ab')):
            states[name + '-intp-string'] = [('\\\\[' + lbrace + ']', String.Other), (lbrace, String.Other, '#push'), (rbrace, String.Other, '#pop'), include('string-intp-escaped'), ('[\\\\#' + bracecc + ']', String.Other), ('[^\\\\#' + bracecc + ']+', String.Other)]
            states['strings'].append(('%' + lbrace, String.Other, name + '-intp-string'))
            states[name + '-string'] = [('\\\\[\\\\' + bracecc + ']', String.Other), (lbrace, String.Other, '#push'), (rbrace, String.Other, '#pop'), ('[\\\\#' + bracecc + ']', String.Other), ('[^\\\\#' + bracecc + ']+', String.Other)]
            states['strings'].append(('%[wi]' + lbrace, String.Other, name + '-string'))
            states[name + '-regex'] = [('\\\\[\\\\' + bracecc + ']', String.Regex), (lbrace, String.Regex, '#push'), (rbrace + '[imsx]*', String.Regex, '#pop'), include('string-intp'), ('[\\\\#' + bracecc + ']', String.Regex), ('[^\\\\#' + bracecc + ']+', String.Regex)]
            states['strings'].append(('%r' + lbrace, String.Regex, name + '-regex'))
        states['strings'] += [('(%r([\\W_]))((?:\\\\\\2|(?!\\2).)*)(\\2[imsx]*)', intp_regex_callback), ('(%[wi]([\\W_]))((?:\\\\\\2|(?!\\2).)*)(\\2)', intp_string_callback), ('(?<=[-+/*%=<>&!^|~,(])(\\s*)(%([\\t ])(?:(?:\\\\\\3|(?!\\3).)*)\\3)', bygroups(Text, String.Other, None)), ('^(\\s*)(%([\\t ])(?:(?:\\\\\\3|(?!\\3).)*)\\3)', bygroups(Text, String.Other, None)), ('(%([\\[{(<]))((?:\\\\\\2|(?!\\2).)*)(\\2)', intp_string_callback)]
        return states
    tokens = {'root': [('#.*?$', Comment.Single), (words('\n                abstract asm as begin break case do else elsif end ensure extend ifdef if\n                include instance_sizeof next of pointerof private protected rescue return\n                require sizeof super then typeof unless until when while with yield\n            '.split(), suffix='\\b'), Keyword), (words(['true', 'false', 'nil'], suffix='\\b'), Keyword.Constant), ('(module|lib)(\\s+)([a-zA-Z_]\\w*(?:::[a-zA-Z_]\\w*)*)', bygroups(Keyword, Text, Name.Namespace)), ('(def|fun|macro)(\\s+)((?:[a-zA-Z_]\\w*::)*)', bygroups(Keyword, Text, Name.Namespace), 'funcname'), ('def(?=[*%&^`~+-/\\[<>=])', Keyword, 'funcname'), ('(class|struct|union|type|alias|enum)(\\s+)((?:[a-zA-Z_]\\w*::)*)', bygroups(Keyword, Text, Name.Namespace), 'classname'), ('(self|out|uninitialized)\\b|(is_a|responds_to)\\?', Keyword.Pseudo), (words('\n                debugger record pp assert_responds_to spawn parallel\n                getter setter property delegate def_hash def_equals def_equals_and_hash\n                forward_missing_to\n            '.split(), suffix='\\b'), Name.Builtin.Pseudo), ('getter[!?]|property[!?]|__(DIR|FILE|LINE)__\\b', Name.Builtin.Pseudo), (words('\n                Object Value Struct Reference Proc Class Nil Symbol Enum Void\n                Bool Number Int Int8 Int16 Int32 Int64 UInt8 UInt16 UInt32 UInt64\n                Float Float32 Float64 Char String\n                Pointer Slice Range Exception Regex\n                Mutex StaticArray Array Hash Set Tuple Deque Box Process File\n                Dir Time Channel Concurrent Scheduler\n                abort at_exit caller delay exit fork future get_stack_top gets\n                lazy loop main p print printf puts\n                raise rand read_line sleep sprintf system with_color\n            '.split(), prefix='(?<!\\.)', suffix='\\b'), Name.Builtin), ('(?<!\\w)(<<-?)(["`\\\']?)([a-zA-Z_]\\w*)(\\2)(.*?\\n)', heredoc_callback), ('(<<-?)("|\\\')()(\\2)(.*?\\n)', heredoc_callback), ('__END__', Comment.Preproc, 'end-part'), ('(?:^|(?<=[=<>~!:])|(?<=(?:\\s|;)when\\s)|(?<=(?:\\s|;)or\\s)|(?<=(?:\\s|;)and\\s)|(?<=\\.index\\s)|(?<=\\.scan\\s)|(?<=\\.sub\\s)|(?<=\\.sub!\\s)|(?<=\\.gsub\\s)|(?<=\\.gsub!\\s)|(?<=\\.match\\s)|(?<=(?:\\s|;)if\\s)|(?<=(?:\\s|;)elsif\\s)|(?<=^when\\s)|(?<=^index\\s)|(?<=^scan\\s)|(?<=^sub\\s)|(?<=^gsub\\s)|(?<=^sub!\\s)|(?<=^gsub!\\s)|(?<=^match\\s)|(?<=^if\\s)|(?<=^elsif\\s))(\\s*)(/)', bygroups(Text, String.Regex), 'multiline-regex'), ('(?<=\\(|,|\\[)/', String.Regex, 'multiline-regex'), ('(\\s+)(/)(?![\\s=])', bygroups(Text, String.Regex), 'multiline-regex'), ('(0o[0-7]+(?:_[0-7]+)*(?:_?[iu][0-9]+)?)\\b(\\s*)([/?])?', bygroups(Number.Oct, Text, Operator)), ('(0x[0-9A-Fa-f]+(?:_[0-9A-Fa-f]+)*(?:_?[iu][0-9]+)?)\\b(\\s*)([/?])?', bygroups(Number.Hex, Text, Operator)), ('(0b[01]+(?:_[01]+)*(?:_?[iu][0-9]+)?)\\b(\\s*)([/?])?', bygroups(Number.Bin, Text, Operator)), ('((?:0(?![0-9])|[1-9][\\d_]*)(?:\\.\\d[\\d_]*)(?:e[+-]?[0-9]+)?(?:_?f[0-9]+)?)(\\s*)([/?])?', bygroups(Number.Float, Text, Operator)), ('((?:0(?![0-9])|[1-9][\\d_]*)(?:\\.\\d[\\d_]*)?(?:e[+-]?[0-9]+)(?:_?f[0-9]+)?)(\\s*)([/?])?', bygroups(Number.Float, Text, Operator)), ('((?:0(?![0-9])|[1-9][\\d_]*)(?:\\.\\d[\\d_]*)?(?:e[+-]?[0-9]+)?(?:_?f[0-9]+))(\\s*)([/?])?', bygroups(Number.Float, Text, Operator)), ('(0\\b|[1-9][\\d]*(?:_\\d+)*(?:_?[iu][0-9]+)?)\\b(\\s*)([/?])?', bygroups(Number.Integer, Text, Operator)), ('@@[a-zA-Z_]\\w*', Name.Variable.Class), ('@[a-zA-Z_]\\w*', Name.Variable.Instance), ('\\$\\w+', Name.Variable.Global), ('\\$[!@&`\\\'+~=/\\\\,;.<>_*$?:"^-]', Name.Variable.Global), ('\\$-[0adFiIlpvw]', Name.Variable.Global), ('::', Operator), include('strings'), ('\\?(\\\\[MC]-)*(\\\\([\\\\befnrtv#"\\\']|x[a-fA-F0-9]{1,2}|[0-7]{1,3})|\\S)(?!\\w)', String.Char), ('[A-Z][A-Z_]+\\b', Name.Constant), ('\\{%', String.Interpol, 'in-macro-control'), ('\\{\\{', String.Interpol, 'in-macro-expr'), ('(@\\[)(\\s*)([A-Z]\\w*)', bygroups(Operator, Text, Name.Decorator), 'in-attr'), (words(CRYSTAL_OPERATORS, prefix='(\\.|::)'), bygroups(Operator, Name.Operator)), ('(\\.|::)([a-zA-Z_]\\w*[!?]?|[*%&^`~+\\-/\\[<>=])', bygroups(Operator, Name)), ('[a-zA-Z_]\\w*(?:[!?](?!=))?', Name), ('(\\[|\\]\\??|\\*\\*|<=>?|>=|<<?|>>?|=~|===|!~|&&?|\\|\\||\\.{1,3})', Operator), ('[-+/*%=<>&!^|~]=?', Operator), ('[(){};,/?:\\\\]', Punctuation), ('\\s+', Text)], 'funcname': [('(?:([a-zA-Z_]\\w*)(\\.))?([a-zA-Z_]\\w*[!?]?|\\*\\*?|[-+]@?|[/%&|^`~]|\\[\\]=?|<<|>>|<=?>|>=?|===?)', bygroups(Name.Class, Operator, Name.Function), '#pop'), default('#pop')], 'classname': [('[A-Z_]\\w*', Name.Class), ('(\\()(\\s*)([A-Z_]\\w*)(\\s*)(\\))', bygroups(Punctuation, Text, Name.Class, Text, Punctuation)), default('#pop')], 'in-intp': [('\\{', String.Interpol, '#push'), ('\\}', String.Interpol, '#pop'), include('root')], 'string-intp': [('#\\{', String.Interpol, 'in-intp')], 'string-escaped': [('\\\\([\\\\befnstv#"\\\']|x[a-fA-F0-9]{1,2}|[0-7]{1,3})', String.Escape)], 'string-intp-escaped': [include('string-intp'), include('string-escaped')], 'interpolated-regex': [include('string-intp'), ('[\\\\#]', String.Regex), ('[^\\\\#]+', String.Regex)], 'interpolated-string': [include('string-intp'), ('[\\\\#]', String.Other), ('[^\\\\#]+', String.Other)], 'multiline-regex': [include('string-intp'), ('\\\\\\\\', String.Regex), ('\\\\/', String.Regex), ('[\\\\#]', String.Regex), ('[^\\\\/#]+', String.Regex), ('/[imsx]*', String.Regex, '#pop')], 'end-part': [('.+', Comment.Preproc, '#pop')], 'in-macro-control': [('\\{%', String.Interpol, '#push'), ('%\\}', String.Interpol, '#pop'), ('for\\b|in\\b', Keyword), include('root')], 'in-macro-expr': [('\\{\\{', String.Interpol, '#push'), ('\\}\\}', String.Interpol, '#pop'), include('root')], 'in-attr': [('\\[', Operator, '#push'), ('\\]', Operator, '#pop'), include('root')]}
    tokens.update(gen_crystalstrings_rules())