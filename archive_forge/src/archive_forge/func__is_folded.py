import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
@staticmethod
def _is_folded(items):
    if len(items) < 2:
        return False
    for i in items:
        if not isinstance(i, Character) or not i.positive or (not i.case_flags):
            return False
    folded = ''.join((chr(i.value) for i in items))
    folded = _regex.fold_case(FULL_CASE_FOLDING, folded)
    expanding_chars = _regex.get_expand_on_folding()
    for c in expanding_chars:
        if folded == _regex.fold_case(FULL_CASE_FOLDING, c):
            return True
    return False