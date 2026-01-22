from __future__ import annotations
import abc
import logging
import os
import sys
import typing
import warnings
from urwid import signals
from urwid.util import StoppingContext, int_scale
def _parse_color_true(desc: str) -> int | None:
    c = _parse_color_256(desc)
    if c is not None:
        r, g, b = _COLOR_VALUES_256[c]
        return (r << 16) + (g << 8) + b
    if not desc.startswith('#'):
        return None
    if len(desc) == 7:
        h = desc[1:]
        return int(h, 16)
    if len(desc) == 4:
        h = f'0x{desc[1]}0{desc[2]}0{desc[3]}'
        return int(h, 16)
    return None