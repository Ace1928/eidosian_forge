from __future__ import annotations
import abc
import logging
import os
import sys
import typing
import warnings
from urwid import signals
from urwid.util import StoppingContext, int_scale
def _color_desc_true(num: int) -> str:
    return f'#{num:06x}'