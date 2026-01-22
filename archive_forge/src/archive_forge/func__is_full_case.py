import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
@staticmethod
def _is_full_case(items, i):
    if not 0 <= i < len(items):
        return False
    item = items[i]
    return isinstance(item, Character) and item.positive and (item.case_flags & FULLIGNORECASE == FULLIGNORECASE)