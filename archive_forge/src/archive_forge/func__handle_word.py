from __future__ import annotations
from typing import TYPE_CHECKING
from . import Extension
from ..treeprocessors import Treeprocessor
import re
def _handle_word(s, t):
    if t.startswith('.'):
        return ('.', t[1:])
    if t.startswith('#'):
        return ('id', t[1:])
    return (t, t)