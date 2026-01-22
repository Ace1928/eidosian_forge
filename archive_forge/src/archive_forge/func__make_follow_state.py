import re
from pygments.lexer import Lexer, RegexLexer, do_insertions, bygroups, \
from pygments.token import Punctuation, \
from pygments.util import shebang_matches
def _make_follow_state(compound, _label=_label, _label_compound=_label_compound, _nl=_nl, _space=_space, _start_label=_start_label, _token=_token, _token_compound=_token_compound, _ws=_ws):
    suffix = '/compound' if compound else ''
    state = []
    if compound:
        state.append(('(?=\\))', Text, '#pop'))
    state += [('%s([%s]*)(%s)(.*)' % (_start_label, _ws, _label_compound if compound else _label), bygroups(Text, Punctuation, Text, Name.Label, Comment.Single)), include('redirect%s' % suffix), ('(?=[%s])' % _nl, Text, '#pop'), ('\\|\\|?|&&?', Punctuation, '#pop'), include('text')]
    return state