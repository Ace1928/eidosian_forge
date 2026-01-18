from __future__ import annotations
import re
from collections.abc import Generator
from typing import NamedTuple

    Tokenize JavaScript/JSX source.  Returns a generator of tokens.

    :param jsx: Enable (limited) JSX parsing.
    :param dotted: Read dotted names as single name token.
    :param template_string: Support ES6 template strings
    :param lineno: starting line number (optional)
    