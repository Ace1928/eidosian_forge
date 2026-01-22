import copy
import io
from collections import (
from enum import (
from typing import (
from wcwidth import (  # type: ignore[import]
from . import (
@staticmethod
def _wrap_long_word(word: str, max_width: int, max_lines: Union[int, float], is_last_word: bool) -> Tuple[str, int, int]:
    """
        Used by _wrap_text() to wrap a long word over multiple lines

        :param word: word being wrapped
        :param max_width: maximum display width of a line
        :param max_lines: maximum lines to wrap before ending the last line displayed with an ellipsis
        :param is_last_word: True if this is the last word of the total text being wrapped
        :return: Tuple(wrapped text, lines used, display width of last line)
        """
    styles_dict = utils.get_styles_dict(word)
    wrapped_buf = io.StringIO()
    total_lines = 1
    cur_line_width = 0
    char_index = 0
    while char_index < len(word):
        if total_lines == max_lines:
            remaining_word = word[char_index:]
            if not is_last_word and ansi.style_aware_wcswidth(remaining_word) == max_width:
                remaining_word += 'EXTRA'
            truncated_line = utils.truncate_line(remaining_word, max_width)
            cur_line_width = ansi.style_aware_wcswidth(truncated_line)
            wrapped_buf.write(truncated_line)
            break
        if char_index in styles_dict:
            wrapped_buf.write(styles_dict[char_index])
            char_index += len(styles_dict[char_index])
            continue
        cur_char = word[char_index]
        cur_char_width = wcwidth(cur_char)
        if cur_char_width > max_width:
            cur_char = constants.HORIZONTAL_ELLIPSIS
            cur_char_width = wcwidth(cur_char)
        if cur_line_width + cur_char_width > max_width:
            wrapped_buf.write('\n')
            total_lines += 1
            cur_line_width = 0
            continue
        cur_line_width += cur_char_width
        wrapped_buf.write(cur_char)
        char_index += 1
    return (wrapped_buf.getvalue(), total_lines, cur_line_width)