from __future__ import print_function
import re
import sys
import time
from pygments.filter import apply_filters, Filter
from pygments.filters import get_filter_by_name
from pygments.token import Error, Text, Other, _TokenType
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
from pygments.regexopt import regex_opt
def _process_new_state(cls, new_state, unprocessed, processed):
    """Preprocess the state transition action of a token definition."""
    if isinstance(new_state, str):
        if new_state == '#pop':
            return -1
        elif new_state in unprocessed:
            return (new_state,)
        elif new_state == '#push':
            return new_state
        elif new_state[:5] == '#pop:':
            return -int(new_state[5:])
        else:
            assert False, 'unknown new state %r' % new_state
    elif isinstance(new_state, combined):
        tmp_state = '_tmp_%d' % cls._tmpname
        cls._tmpname += 1
        itokens = []
        for istate in new_state:
            assert istate != new_state, 'circular state ref %r' % istate
            itokens.extend(cls._process_state(unprocessed, processed, istate))
        processed[tmp_state] = itokens
        return (tmp_state,)
    elif isinstance(new_state, tuple):
        for istate in new_state:
            assert istate in unprocessed or istate in ('#pop', '#push'), 'unknown new state ' + istate
        return new_state
    else:
        assert False, 'unknown new state def %r' % new_state