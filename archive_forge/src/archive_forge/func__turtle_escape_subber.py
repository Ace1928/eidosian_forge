import codecs
import re
import warnings
from typing import Match
def _turtle_escape_subber(match: Match[str]) -> str:
    smatch, umatch = match.groups()
    if smatch is not None:
        return _string_escape_map[smatch]
    else:
        return chr(int(umatch[1:], 16))