from __future__ import annotations
import codecs
import re
import sys
import typing as t
from collections import deque
from traceback import format_exception_only
from markupsafe import escape
def _add_subclass_info(inner: str, obj: object, base: t.Type | tuple[t.Type, ...]) -> str:
    if isinstance(base, tuple):
        for cls in base:
            if type(obj) is cls:
                return inner
    elif type(obj) is base:
        return inner
    module = ''
    if obj.__class__.__module__ not in ('__builtin__', 'exceptions'):
        module = f'<span class="module">{obj.__class__.__module__}.</span>'
    return f'{module}{type(obj).__name__}({inner})'