from __future__ import annotations
import abc
import logging
import os
import sys
import typing
import warnings
from urwid import signals
from urwid.util import StoppingContext, int_scale
def _true_to_256(desc: str) -> str | None:
    if not (desc.startswith('#') and len(desc) == 7):
        return None
    c256 = _parse_color_256('#' + ''.join((format(int(x, 16) // 16, 'x') for x in (desc[1:3], desc[3:5], desc[5:7]))))
    return _color_desc_256(c256)