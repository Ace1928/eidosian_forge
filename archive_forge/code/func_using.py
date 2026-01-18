from __future__ import print_function
import re
import sys
import time
from pygments.filter import apply_filters, Filter
from pygments.filters import get_filter_by_name
from pygments.token import Error, Text, Other, _TokenType
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
from pygments.regexopt import regex_opt
def using(_other, **kwargs):
    """
    Callback that processes the match with a different lexer.

    The keyword arguments are forwarded to the lexer, except `state` which
    is handled separately.

    `state` specifies the state that the new lexer will start in, and can
    be an enumerable such as ('root', 'inline', 'string') or a simple
    string which is assumed to be on top of the root state.

    Note: For that to work, `_other` must not be an `ExtendedRegexLexer`.
    """
    gt_kwargs = {}
    if 'state' in kwargs:
        s = kwargs.pop('state')
        if isinstance(s, (list, tuple)):
            gt_kwargs['stack'] = s
        else:
            gt_kwargs['stack'] = ('root', s)
    if _other is this:

        def callback(lexer, match, ctx=None):
            if kwargs:
                kwargs.update(lexer.options)
                lx = lexer.__class__(**kwargs)
            else:
                lx = lexer
            s = match.start()
            for i, t, v in lx.get_tokens_unprocessed(match.group(), **gt_kwargs):
                yield (i + s, t, v)
            if ctx:
                ctx.pos = match.end()
    else:

        def callback(lexer, match, ctx=None):
            kwargs.update(lexer.options)
            lx = _other(**kwargs)
            s = match.start()
            for i, t, v in lx.get_tokens_unprocessed(match.group(), **gt_kwargs):
                yield (i + s, t, v)
            if ctx:
                ctx.pos = match.end()
    return callback