from __future__ import annotations
import re
import sys
from types import TracebackType
from typing import Any
import pytest
import trio
from trio.testing import Matcher, RaisesGroup
def wrap_escape(s: str) -> str:
    return '^' + re.escape(s) + '$'