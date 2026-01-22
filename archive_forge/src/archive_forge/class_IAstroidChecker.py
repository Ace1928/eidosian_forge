from __future__ import annotations
import os
import typing as t
import astroid
from pylint.checkers import BaseChecker
class IAstroidChecker:
    """Backwards compatibility for 2.x / 3.x support."""