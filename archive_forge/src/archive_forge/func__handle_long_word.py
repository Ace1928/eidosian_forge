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
def _handle_long_word(self, reversed_chunks, cur_line, cur_len, width):
    """_handle_long_word(chunks : [string],
                             cur_line : [string],
                             cur_len : int, width : int)
        Handle a chunk of text (most likely a word, not whitespace) that
        is too long to fit in any line.
        """
    if width < 1:
        space_left = 1
    else:
        space_left = width - cur_len
    if self.break_long_words:
        chunk = reversed_chunks[-1]
        i = 1
        while self._len(chunk[:i]) <= space_left:
            i = i + 1
        cur_line.append(chunk[:i - 1])
        reversed_chunks[-1] = chunk[i - 1:]
    elif not cur_line:
        cur_line.append(reversed_chunks.pop())