from __future__ import annotations
import re
import typing
import warnings
import wcwidth
Return whether pos is within a double-byte encoded character.

    text -- byte string in question
    line_start -- offset of beginning of line (< pos)
    pos -- offset in question

    Return values:
    0 -- not within dbe char, or double_byte_encoding == False
    1 -- pos is on the 1st half of a dbe char
    2 -- pos is on the 2nd half of a dbe char
    