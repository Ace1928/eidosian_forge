import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
@staticmethod
def _flush_char_prefix(info, reverse, prefixed, order, new_branches):
    if not prefixed:
        return
    for value, branches in sorted(prefixed.items(), key=lambda pair: order[pair[0]]):
        if len(branches) == 1:
            new_branches.append(make_sequence(branches[0]))
        else:
            subbranches = []
            optional = False
            for b in branches:
                if len(b) > 1:
                    subbranches.append(make_sequence(b[1:]))
                elif not optional:
                    subbranches.append(Sequence())
                    optional = True
            sequence = Sequence([Character(value), Branch(subbranches)])
            new_branches.append(sequence.optimise(info, reverse))
    prefixed.clear()
    order.clear()