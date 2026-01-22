import logging
import re
from typing import (
from . import settings
from .utils import choplist
def _parse_hexstring(self, s: bytes, i: int) -> int:
    m = END_HEX_STRING.search(s, i)
    if not m:
        self._curtoken += s[i:]
        return len(s)
    j = m.start(0)
    self._curtoken += s[i:j]
    token = HEX_PAIR.sub(lambda m: bytes((int(m.group(0), 16),)), SPC.sub(b'', self._curtoken))
    self._add_token(token)
    self._parse1 = self._parse_main
    return j