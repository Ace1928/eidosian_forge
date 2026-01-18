from __future__ import annotations
import codecs
import re
import sys
import typing as t
from collections import deque
from traceback import format_exception_only
from markupsafe import escape
Displays an HTML version of the normal help, for the interactive
    debugger only because it requires a patched sys.stdout.
    