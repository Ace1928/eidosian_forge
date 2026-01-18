from ..lazyre import LazyReCompile
import inspect
from ..line import cursor_on_closing_char_pair
@edit_keys.on('<Esc+c>')
def titlecase_next_word(cursor_offset, line):
    return (cursor_offset, line)