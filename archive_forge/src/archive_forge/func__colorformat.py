from .base import Style, DEFAULT_ATTRS, ANSI_COLOR_NAMES
from .defaults import DEFAULT_STYLE_EXTENSIONS
from .utils import merge_attrs, split_token_in_parts
from six.moves import range
def _colorformat(text):
    """
    Parse/validate color format.

    Like in Pygments, but also support the ANSI color names.
    (These will map to the colors of the 16 color palette.)
    """
    if text[0:1] == '#':
        col = text[1:]
        if col in ANSI_COLOR_NAMES:
            return col
        elif len(col) == 6:
            return col
        elif len(col) == 3:
            return col[0] * 2 + col[1] * 2 + col[2] * 2
    elif text == '':
        return text
    raise ValueError('Wrong color format %r' % text)