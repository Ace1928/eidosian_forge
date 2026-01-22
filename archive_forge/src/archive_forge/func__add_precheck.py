import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def _add_precheck(self, info, reverse, branches):
    charset = set()
    pos = -1 if reverse else 0
    for branch in branches:
        if type(branch) is Literal and branch.case_flags == NOCASE:
            charset.add(branch.characters[pos])
        else:
            return
    if not charset:
        return None
    return _check_firstset(info, reverse, [Character(c) for c in charset])