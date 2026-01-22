from __future__ import (absolute_import, division, print_function)
def _get_quote_state(token, quote_char):
    """
    the goal of this block is to determine if the quoted string
    is unterminated in which case it needs to be put back together
    """
    prev_char = None
    for idx, cur_char in enumerate(token):
        if idx > 0:
            prev_char = token[idx - 1]
        if cur_char in '"\'' and prev_char != '\\':
            if quote_char:
                if cur_char == quote_char:
                    quote_char = None
            else:
                quote_char = cur_char
    return quote_char