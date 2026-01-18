from __future__ import unicode_literals
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.token import Token
def token_list_width(tokenlist):
    """
    Return the character width of this token list.
    (Take double width characters into account.)

    :param tokenlist: List of (token, text) or (token, text, mouse_handler)
                      tuples.
    """
    ZeroWidthEscape = Token.ZeroWidthEscape
    return sum((get_cwidth(c) for item in tokenlist for c in item[1] if item[0] != ZeroWidthEscape))