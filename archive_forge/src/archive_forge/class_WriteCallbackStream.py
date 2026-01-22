from __future__ import annotations
import ast
import base64
import copy
import io
import pathlib
import pkgutil
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from html import escape
from textwrap import dedent
from typing import Any, Dict, List
import markdown
class WriteCallbackStream(io.StringIO):

    def __init__(self, on_write=None, escape=True):
        self._onwrite = on_write
        self._escape = escape
        super().__init__()

    def write(self, s):
        if self._onwrite:
            self._onwrite(escape(s) if self._escape else s)
        super().write(s)