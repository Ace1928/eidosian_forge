from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import formatting_windows  # pylint: disable=unused-import
import termcolor
def EllipsisMiddleTruncate(text, available_space, line_length):
    """Truncates text from the middle with ellipsis."""
    if available_space < len(ELLIPSIS):
        available_space = line_length
    if len(text) < available_space:
        return text
    available_string_len = available_space - len(ELLIPSIS)
    first_half_len = int(available_string_len / 2)
    second_half_len = available_string_len - first_half_len
    return text[:first_half_len] + ELLIPSIS + text[-second_half_len:]