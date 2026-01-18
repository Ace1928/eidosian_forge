from ..lazyre import LazyReCompile
import inspect
from ..line import cursor_on_closing_char_pair
@edit_keys.on(config='transpose_chars_key')
def transpose_character_before_cursor(cursor_offset, line):
    if cursor_offset < 2:
        return (cursor_offset, line)
    if cursor_offset == len(line):
        return (cursor_offset, line[:-2] + line[-1] + line[-2])
    return (min(len(line), cursor_offset + 1), line[:cursor_offset - 1] + (line[cursor_offset] if len(line) > cursor_offset else '') + line[cursor_offset - 1] + line[cursor_offset + 1:])