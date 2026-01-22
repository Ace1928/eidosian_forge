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
class _CustomTextWrap(textwrap.TextWrapper):
    """A custom implementation of CPython's textwrap.TextWrapper. This supports
    both wide characters (Korea, Japanese, Chinese)  - including mixed string.
    For the most part, the `_handle_long_word` and `_wrap_chunks` functions were
    copy pasted out of the CPython baseline, and updated with our custom length
    and line appending logic.
    """

    def __init__(self, *args, **kwargs):
        self._active_codes = []
        self.max_lines = None
        textwrap.TextWrapper.__init__(self, *args, **kwargs)

    @staticmethod
    def _len(item):
        """Custom len that gets console column width for wide
        and non-wide characters as well as ignores color codes"""
        stripped = _strip_ansi(item)
        if wcwidth:
            return wcwidth.wcswidth(stripped)
        else:
            return len(stripped)

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

    def _wrap_chunks(self, chunks):
        """_wrap_chunks(chunks : [string]) -> [string]
        Wrap a sequence of text chunks and return a list of lines of
        length 'self.width' or less.  (If 'break_long_words' is false,
        some lines may be longer than this.)  Chunks correspond roughly
        to words and the whitespace between them: each chunk is
        indivisible (modulo 'break_long_words'), but a line break can
        come between any two chunks.  Chunks should not have internal
        whitespace; ie. a chunk is either all whitespace or a "word".
        Whitespace chunks will be removed from the beginning and end of
        lines, but apart from that whitespace is preserved.
        """
        lines = []
        if self.width <= 0:
            raise ValueError('invalid width %r (must be > 0)' % self.width)
        if self.max_lines is not None:
            if self.max_lines > 1:
                indent = self.subsequent_indent
            else:
                indent = self.initial_indent
            if self._len(indent) + self._len(self.placeholder.lstrip()) > self.width:
                raise ValueError('placeholder too large for max width')
        chunks.reverse()
        while chunks:
            cur_line = []
            cur_len = 0
            if lines:
                indent = self.subsequent_indent
            else:
                indent = self.initial_indent
            width = self.width - self._len(indent)
            if self.drop_whitespace and chunks[-1].strip() == '' and lines:
                del chunks[-1]
            while chunks:
                chunk_len = self._len(chunks[-1])
                if cur_len + chunk_len <= width:
                    cur_line.append(chunks.pop())
                    cur_len += chunk_len
                else:
                    break
            if chunks and self._len(chunks[-1]) > width:
                self._handle_long_word(chunks, cur_line, cur_len, width)
                cur_len = sum(map(self._len, cur_line))
            if self.drop_whitespace and cur_line and (cur_line[-1].strip() == ''):
                cur_len -= self._len(cur_line[-1])
                del cur_line[-1]
            if cur_line:
                if self.max_lines is None or len(lines) + 1 < self.max_lines or ((not chunks or (self.drop_whitespace and len(chunks) == 1 and (not chunks[0].strip()))) and cur_len <= width):
                    self._update_lines(lines, indent + ''.join(cur_line))
                else:
                    while cur_line:
                        if cur_line[-1].strip() and cur_len + self._len(self.placeholder) <= width:
                            cur_line.append(self.placeholder)
                            self._update_lines(lines, indent + ''.join(cur_line))
                            break
                        cur_len -= self._len(cur_line[-1])
                        del cur_line[-1]
                    else:
                        if lines:
                            prev_line = lines[-1].rstrip()
                            if self._len(prev_line) + self._len(self.placeholder) <= self.width:
                                lines[-1] = prev_line + self.placeholder
                                break
                        self._update_lines(lines, indent + self.placeholder.lstrip())
                    break
        return lines