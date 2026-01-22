from __future__ import annotations
import io
import os
import sys
import warnings
from typing import Any, Iterable
from ._types import Attribute, Color, Highlight
def _can_do_colour(*, no_color: bool | None=None, force_color: bool | None=None) -> bool:
    """Check env vars and for tty/dumb terminal"""
    if no_color is not None and no_color:
        return False
    if force_color is not None and force_color:
        return True
    if 'ANSI_COLORS_DISABLED' in os.environ:
        return False
    if 'NO_COLOR' in os.environ:
        return False
    if 'FORCE_COLOR' in os.environ:
        return True
    if os.environ.get('TERM') == 'dumb':
        return False
    if not hasattr(sys.stdout, 'fileno'):
        return False
    try:
        return os.isatty(sys.stdout.fileno())
    except io.UnsupportedOperation:
        return sys.stdout.isatty()