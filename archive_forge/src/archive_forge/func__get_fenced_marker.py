import re
from typing import Dict, Any
from textwrap import indent
from ._list import render_list
from ..core import BaseRenderer, BlockState
from ..util import strip_end
def _get_fenced_marker(code):
    found = fenced_re.findall(code)
    if not found:
        return '```'
    ticks = []
    waves = []
    for s in found:
        if s[0] == '`':
            ticks.append(len(s))
        else:
            waves.append(len(s))
    if not ticks:
        return '```'
    if not waves:
        return '~~~'
    return '`' * (max(ticks) + 1)