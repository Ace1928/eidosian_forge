from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import formatting_windows  # pylint: disable=unused-import
import termcolor
def EllipsisTruncate(text, available_space, line_length):
    """Truncate text from the end with ellipsis."""
    if available_space < len(ELLIPSIS):
        available_space = line_length
    if len(text) <= available_space:
        return text
    return text[:available_space - len(ELLIPSIS)] + ELLIPSIS