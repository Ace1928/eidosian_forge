from __future__ import annotations
from typing import TYPE_CHECKING
from . import Extension
from ..treeprocessors import Treeprocessor
import re
def _handle_key_value(s, t):
    return t.split('=', 1)