from __future__ import annotations
import functools
import typing
from urwid.str_util import calc_text_pos, calc_width, get_char_width, is_wide_char, move_next_char, move_prev_char
from urwid.util import calc_trim_text, get_encoding

        Return a "sub-segment" list containing segment structures
        that make up a portion of this segment.

        A list is returned to handle cases where wide characters
        need to be replaced with a space character at either edge
        so two or three segments will be returned.
        