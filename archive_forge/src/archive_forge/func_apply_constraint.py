import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def apply_constraint(source, info, constraints, case_flags, saved_pos, sequence):
    element = sequence.pop()
    if element is None:
        raise error('nothing for fuzzy constraint', source.string, saved_pos)
    if isinstance(element, Group):
        element.subpattern = Fuzzy(element.subpattern, constraints)
        sequence.append(element)
    else:
        sequence.append(Fuzzy(element, constraints))