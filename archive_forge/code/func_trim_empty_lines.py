import numbers
import random
import re
import string
import textwrap
def trim_empty_lines(text):
    """
    Trim leading and trailing empty lines from the given text.

    :param text: The text to trim (a string).
    :returns: The trimmed text (a string).
    """
    lines = text.splitlines(True)
    while lines and is_empty_line(lines[0]):
        lines.pop(0)
    while lines and is_empty_line(lines[-1]):
        lines.pop(-1)
    return ''.join(lines)