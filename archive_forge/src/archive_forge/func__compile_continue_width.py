import re
from .core import BlockState
from .util import (
def _compile_continue_width(text, leading_width):
    text = expand_leading_tab(text, 3)
    text = expand_tab(text)
    m2 = _LINE_HAS_TEXT.match(text)
    if m2:
        if text.startswith('     '):
            space_width = 1
        else:
            space_width = len(m2.group(1))
        text = text[space_width:] + '\n'
    else:
        space_width = 1
        text = ''
    continue_width = leading_width + space_width
    return (text, continue_width)