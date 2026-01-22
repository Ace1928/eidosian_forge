import re
from pygments.lexer import Lexer, RegexLexer, do_insertions, bygroups, \
from pygments.token import Punctuation, \
from pygments.util import shebang_matches
class BatchLexer(RegexLexer):
    """
    Lexer for the DOS/Windows Batch file format.

    .. versionadded:: 0.7
    """
    name = 'Batchfile'
    aliases = ['bat', 'batch', 'dosbatch', 'winbatch']
    filenames = ['*.bat', '*.cmd']
    mimetypes = ['application/x-dos-batch']
    flags = re.MULTILINE | re.IGNORECASE
    _nl = '\\n\\x1a'
    _punct = '&<>|'
    _ws = '\\t\\v\\f\\r ,;=\\xa0'
    _space = '(?:(?:(?:\\^[%s])?[%s])+)' % (_nl, _ws)
    _keyword_terminator = '(?=(?:\\^[%s]?)?[%s+./:[\\\\\\]]|[%s%s(])' % (_nl, _ws, _nl, _punct)
    _token_terminator = '(?=\\^?[%s]|[%s%s])' % (_ws, _punct, _nl)
    _start_label = '((?:(?<=^[^:])|^[^:]?)[%s]*)(:)' % _ws
    _label = '(?:(?:[^%s%s%s+:^]|\\^[%s]?[\\w\\W])*)' % (_nl, _punct, _ws, _nl)
    _label_compound = '(?:(?:[^%s%s%s+:^)]|\\^[%s]?[^)])*)' % (_nl, _punct, _ws, _nl)
    _number = '(?:-?(?:0[0-7]+|0x[\\da-f]+|\\d+)%s)' % _token_terminator
    _opword = '(?:equ|geq|gtr|leq|lss|neq)'
    _string = '(?:"[^%s"]*(?:"|(?=[%s])))' % (_nl, _nl)
    _variable = '(?:(?:%%(?:\\*|(?:~[a-z]*(?:\\$[^:]+:)?)?\\d|[^%%:%s]+(?::(?:~(?:-?\\d+)?(?:,(?:-?\\d+)?)?|(?:[^%%%s^]|\\^[^%%%s])[^=%s]*=(?:[^%%%s^]|\\^[^%%%s])*)?)?%%))|(?:\\^?![^!:%s]+(?::(?:~(?:-?\\d+)?(?:,(?:-?\\d+)?)?|(?:[^!%s^]|\\^[^!%s])[^=%s]*=(?:[^!%s^]|\\^[^!%s])*)?)?\\^?!))' % (_nl, _nl, _nl, _nl, _nl, _nl, _nl, _nl, _nl, _nl, _nl, _nl)
    _core_token = '(?:(?:(?:\\^[%s]?)?[^"%s%s%s])+)' % (_nl, _nl, _punct, _ws)
    _core_token_compound = '(?:(?:(?:\\^[%s]?)?[^"%s%s%s)])+)' % (_nl, _nl, _punct, _ws)
    _token = '(?:[%s]+|%s)' % (_punct, _core_token)
    _token_compound = '(?:[%s]+|%s)' % (_punct, _core_token_compound)
    _stoken = '(?:[%s]+|(?:%s|%s|%s)+)' % (_punct, _string, _variable, _core_token)

    def _make_begin_state(compound, _core_token=_core_token, _core_token_compound=_core_token_compound, _keyword_terminator=_keyword_terminator, _nl=_nl, _punct=_punct, _string=_string, _space=_space, _start_label=_start_label, _stoken=_stoken, _token_terminator=_token_terminator, _variable=_variable, _ws=_ws):
        rest = '(?:%s|%s|[^"%%%s%s%s])*' % (_string, _variable, _nl, _punct, ')' if compound else '')
        rest_of_line = '(?:(?:[^%s^]|\\^[%s]?[\\w\\W])*)' % (_nl, _nl)
        rest_of_line_compound = '(?:(?:[^%s^)]|\\^[%s]?[^)])*)' % (_nl, _nl)
        set_space = '((?:(?:\\^[%s]?)?[^\\S\\n])*)' % _nl
        suffix = ''
        if compound:
            _keyword_terminator = '(?:(?=\\))|%s)' % _keyword_terminator
            _token_terminator = '(?:(?=\\))|%s)' % _token_terminator
            suffix = '/compound'
        return [('\\)', Punctuation, '#pop') if compound else ('\\)((?=\\()|%s)%s' % (_token_terminator, rest_of_line), Comment.Single), ('(?=%s)' % _start_label, Text, 'follow%s' % suffix), (_space, using(this, state='text')), include('redirect%s' % suffix), ('[%s]+' % _nl, Text), ('\\(', Punctuation, 'root/compound'), ('@+', Punctuation), ('((?:for|if|rem)(?:(?=(?:\\^[%s]?)?/)|(?:(?!\\^)|(?<=m))(?:(?=\\()|%s)))(%s?%s?(?:\\^[%s]?)?/(?:\\^[%s]?)?\\?)' % (_nl, _token_terminator, _space, _core_token_compound if compound else _core_token, _nl, _nl), bygroups(Keyword, using(this, state='text')), 'follow%s' % suffix), ('(goto%s)(%s(?:\\^[%s]?)?/(?:\\^[%s]?)?\\?%s)' % (_keyword_terminator, rest, _nl, _nl, rest), bygroups(Keyword, using(this, state='text')), 'follow%s' % suffix), (words(('assoc', 'break', 'cd', 'chdir', 'cls', 'color', 'copy', 'date', 'del', 'dir', 'dpath', 'echo', 'endlocal', 'erase', 'exit', 'ftype', 'keys', 'md', 'mkdir', 'mklink', 'move', 'path', 'pause', 'popd', 'prompt', 'pushd', 'rd', 'ren', 'rename', 'rmdir', 'setlocal', 'shift', 'start', 'time', 'title', 'type', 'ver', 'verify', 'vol'), suffix=_keyword_terminator), Keyword, 'follow%s' % suffix), ('(call)(%s?)(:)' % _space, bygroups(Keyword, using(this, state='text'), Punctuation), 'call%s' % suffix), ('call%s' % _keyword_terminator, Keyword), ('(for%s(?!\\^))(%s)(/f%s)' % (_token_terminator, _space, _token_terminator), bygroups(Keyword, using(this, state='text'), Keyword), ('for/f', 'for')), ('(for%s(?!\\^))(%s)(/l%s)' % (_token_terminator, _space, _token_terminator), bygroups(Keyword, using(this, state='text'), Keyword), ('for/l', 'for')), ('for%s(?!\\^)' % _token_terminator, Keyword, ('for2', 'for')), ('(goto%s)(%s?)(:?)' % (_keyword_terminator, _space), bygroups(Keyword, using(this, state='text'), Punctuation), 'label%s' % suffix), ('(if(?:(?=\\()|%s)(?!\\^))(%s?)((?:/i%s)?)(%s?)((?:not%s)?)(%s?)' % (_token_terminator, _space, _token_terminator, _space, _token_terminator, _space), bygroups(Keyword, using(this, state='text'), Keyword, using(this, state='text'), Keyword, using(this, state='text')), ('(?', 'if')), ('rem(((?=\\()|%s)%s?%s?.*|%s%s)' % (_token_terminator, _space, _stoken, _keyword_terminator, rest_of_line_compound if compound else rest_of_line), Comment.Single, 'follow%s' % suffix), ('(set%s)%s(/a)' % (_keyword_terminator, set_space), bygroups(Keyword, using(this, state='text'), Keyword), 'arithmetic%s' % suffix), ('(set%s)%s((?:/p)?)%s((?:(?:(?:\\^[%s]?)?[^"%s%s^=%s]|\\^[%s]?[^"=])+)?)((?:(?:\\^[%s]?)?=)?)' % (_keyword_terminator, set_space, set_space, _nl, _nl, _punct, ')' if compound else '', _nl, _nl), bygroups(Keyword, using(this, state='text'), Keyword, using(this, state='text'), using(this, state='variable'), Punctuation), 'follow%s' % suffix), default('follow%s' % suffix)]

    def _make_follow_state(compound, _label=_label, _label_compound=_label_compound, _nl=_nl, _space=_space, _start_label=_start_label, _token=_token, _token_compound=_token_compound, _ws=_ws):
        suffix = '/compound' if compound else ''
        state = []
        if compound:
            state.append(('(?=\\))', Text, '#pop'))
        state += [('%s([%s]*)(%s)(.*)' % (_start_label, _ws, _label_compound if compound else _label), bygroups(Text, Punctuation, Text, Name.Label, Comment.Single)), include('redirect%s' % suffix), ('(?=[%s])' % _nl, Text, '#pop'), ('\\|\\|?|&&?', Punctuation, '#pop'), include('text')]
        return state

    def _make_arithmetic_state(compound, _nl=_nl, _punct=_punct, _string=_string, _variable=_variable, _ws=_ws):
        op = '=+\\-*/!~'
        state = []
        if compound:
            state.append(('(?=\\))', Text, '#pop'))
        state += [('0[0-7]+', Number.Oct), ('0x[\\da-f]+', Number.Hex), ('\\d+', Number.Integer), ('[(),]+', Punctuation), ('([%s]|%%|\\^\\^)+' % op, Operator), ('(%s|%s|(\\^[%s]?)?[^()%s%%^"%s%s%s]|\\^[%s%s]?%s)+' % (_string, _variable, _nl, op, _nl, _punct, _ws, _nl, _ws, '[^)]' if compound else '[\\w\\W]'), using(this, state='variable')), ('(?=[\\x00|&])', Text, '#pop'), include('follow')]
        return state

    def _make_call_state(compound, _label=_label, _label_compound=_label_compound):
        state = []
        if compound:
            state.append(('(?=\\))', Text, '#pop'))
        state.append(('(:?)(%s)' % (_label_compound if compound else _label), bygroups(Punctuation, Name.Label), '#pop'))
        return state

    def _make_label_state(compound, _label=_label, _label_compound=_label_compound, _nl=_nl, _punct=_punct, _string=_string, _variable=_variable):
        state = []
        if compound:
            state.append(('(?=\\))', Text, '#pop'))
        state.append(('(%s?)((?:%s|%s|\\^[%s]?%s|[^"%%^%s%s%s])*)' % (_label_compound if compound else _label, _string, _variable, _nl, '[^)]' if compound else '[\\w\\W]', _nl, _punct, ')' if compound else ''), bygroups(Name.Label, Comment.Single), '#pop'))
        return state

    def _make_redirect_state(compound, _core_token_compound=_core_token_compound, _nl=_nl, _punct=_punct, _stoken=_stoken, _string=_string, _space=_space, _variable=_variable, _ws=_ws):
        stoken_compound = '(?:[%s]+|(?:%s|%s|%s)+)' % (_punct, _string, _variable, _core_token_compound)
        return [('((?:(?<=[%s%s])\\d)?)(>>?&|<&)([%s%s]*)(\\d)' % (_nl, _ws, _nl, _ws), bygroups(Number.Integer, Punctuation, Text, Number.Integer)), ('((?:(?<=[%s%s])(?<!\\^[%s])\\d)?)(>>?|<)(%s?%s)' % (_nl, _ws, _nl, _space, stoken_compound if compound else _stoken), bygroups(Number.Integer, Punctuation, using(this, state='text')))]
    tokens = {'root': _make_begin_state(False), 'follow': _make_follow_state(False), 'arithmetic': _make_arithmetic_state(False), 'call': _make_call_state(False), 'label': _make_label_state(False), 'redirect': _make_redirect_state(False), 'root/compound': _make_begin_state(True), 'follow/compound': _make_follow_state(True), 'arithmetic/compound': _make_arithmetic_state(True), 'call/compound': _make_call_state(True), 'label/compound': _make_label_state(True), 'redirect/compound': _make_redirect_state(True), 'variable-or-escape': [(_variable, Name.Variable), ('%%%%|\\^[%s]?(\\^!|[\\w\\W])' % _nl, String.Escape)], 'string': [('"', String.Double, '#pop'), (_variable, Name.Variable), ('\\^!|%%', String.Escape), ('[^"%%^%s]+|[%%^]' % _nl, String.Double), default('#pop')], 'sqstring': [include('variable-or-escape'), ('[^%]+|%', String.Single)], 'bqstring': [include('variable-or-escape'), ('[^%]+|%', String.Backtick)], 'text': [('"', String.Double, 'string'), include('variable-or-escape'), ('[^"%%^%s%s%s\\d)]+|.' % (_nl, _punct, _ws), Text)], 'variable': [('"', String.Double, 'string'), include('variable-or-escape'), ('[^"%%^%s]+|.' % _nl, Name.Variable)], 'for': [('(%s)(in)(%s)(\\()' % (_space, _space), bygroups(using(this, state='text'), Keyword, using(this, state='text'), Punctuation), '#pop'), include('follow')], 'for2': [('\\)', Punctuation), ('(%s)(do%s)' % (_space, _token_terminator), bygroups(using(this, state='text'), Keyword), '#pop'), ('[%s]+' % _nl, Text), include('follow')], 'for/f': [('(")((?:%s|[^"])*?")([%s%s]*)(\\))' % (_variable, _nl, _ws), bygroups(String.Double, using(this, state='string'), Text, Punctuation)), ('"', String.Double, ('#pop', 'for2', 'string')), ("('(?:%%%%|%s|[\\w\\W])*?')([%s%s]*)(\\))" % (_variable, _nl, _ws), bygroups(using(this, state='sqstring'), Text, Punctuation)), ('(`(?:%%%%|%s|[\\w\\W])*?`)([%s%s]*)(\\))' % (_variable, _nl, _ws), bygroups(using(this, state='bqstring'), Text, Punctuation)), include('for2')], 'for/l': [('-?\\d+', Number.Integer), include('for2')], 'if': [('((?:cmdextversion|errorlevel)%s)(%s)(\\d+)' % (_token_terminator, _space), bygroups(Keyword, using(this, state='text'), Number.Integer), '#pop'), ('(defined%s)(%s)(%s)' % (_token_terminator, _space, _stoken), bygroups(Keyword, using(this, state='text'), using(this, state='variable')), '#pop'), ('(exist%s)(%s%s)' % (_token_terminator, _space, _stoken), bygroups(Keyword, using(this, state='text')), '#pop'), ('(%s%s)(%s)(%s%s)' % (_number, _space, _opword, _space, _number), bygroups(using(this, state='arithmetic'), Operator.Word, using(this, state='arithmetic')), '#pop'), (_stoken, using(this, state='text'), ('#pop', 'if2'))], 'if2': [('(%s?)(==)(%s?%s)' % (_space, _space, _stoken), bygroups(using(this, state='text'), Operator, using(this, state='text')), '#pop'), ('(%s)(%s)(%s%s)' % (_space, _opword, _space, _stoken), bygroups(using(this, state='text'), Operator.Word, using(this, state='text')), '#pop')], '(?': [(_space, using(this, state='text')), ('\\(', Punctuation, ('#pop', 'else?', 'root/compound')), default('#pop')], 'else?': [(_space, using(this, state='text')), ('else%s' % _token_terminator, Keyword, '#pop'), default('#pop')]}