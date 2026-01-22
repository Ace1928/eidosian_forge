import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
@staticmethod
def _flush_characters(info, characters, case_flags, items):
    if not characters:
        return
    if case_flags & IGNORECASE:
        if not any((is_cased_i(info, c) for c in characters)):
            case_flags = NOCASE
    if case_flags & FULLIGNORECASE == FULLIGNORECASE:
        literals = Sequence._fix_full_casefold(characters)
        for item in literals:
            chars = item.characters
            if len(chars) == 1:
                items.append(Character(chars[0], case_flags=item.case_flags))
            else:
                items.append(String(chars, case_flags=item.case_flags))
    elif len(characters) == 1:
        items.append(Character(characters[0], case_flags=case_flags))
    else:
        items.append(String(characters, case_flags=case_flags))
    characters[:] = []