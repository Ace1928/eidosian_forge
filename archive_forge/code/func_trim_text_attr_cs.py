from __future__ import annotations
import codecs
import contextlib
import sys
import typing
import warnings
from contextlib import suppress
from urwid import str_util
def trim_text_attr_cs(text: bytes, attr, cs, start_col: int, end_col: int):
    """
    Return ( trimmed text, trimmed attr, trimmed cs ).
    """
    spos, epos, pad_left, pad_right = calc_trim_text(text, 0, len(text), start_col, end_col)
    attrtr = rle_subseg(attr, spos, epos)
    cstr = rle_subseg(cs, spos, epos)
    if pad_left:
        al = rle_get_at(attr, spos - 1)
        rle_prepend_modify(attrtr, (al, 1))
        rle_prepend_modify(cstr, (None, 1))
    if pad_right:
        al = rle_get_at(attr, epos)
        rle_append_modify(attrtr, (al, 1))
        rle_append_modify(cstr, (None, 1))
    return (b''.rjust(pad_left) + text[spos:epos] + b''.rjust(pad_right), attrtr, cstr)