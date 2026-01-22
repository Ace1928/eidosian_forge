from collections import namedtuple
from collections.abc import Iterable, Sized
from html import escape as htmlescape
from itertools import chain, zip_longest as izip_longest
from functools import reduce, partial
import io
import re
import math
import textwrap
import dataclasses
def _update_lines(self, lines, new_line):
    """Adds a new line to the list of lines the text is being wrapped into
        This function will also track any ANSI color codes in this string as well
        as add any colors from previous lines order to preserve the same formatting
        as a single unwrapped string.
        """
    code_matches = [x for x in _ansi_codes.finditer(new_line)]
    color_codes = [code.string[code.span()[0]:code.span()[1]] for code in code_matches]
    new_line = ''.join(self._active_codes) + new_line
    for code in color_codes:
        if code != _ansi_color_reset_code:
            self._active_codes.append(code)
        else:
            self._active_codes = []
    if len(self._active_codes) > 0:
        new_line = new_line + _ansi_color_reset_code
    lines.append(new_line)