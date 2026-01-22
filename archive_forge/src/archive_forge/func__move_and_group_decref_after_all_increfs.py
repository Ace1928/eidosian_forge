import re
from collections import defaultdict, deque
from llvmlite import binding as ll
from numba.core import cgutils
def _move_and_group_decref_after_all_increfs(bb_lines):
    last_incref_pos = 0
    for pos, ln in enumerate(bb_lines):
        if _regex_incref.match(ln) is not None:
            last_incref_pos = pos + 1
    last_decref_pos = 0
    for pos, ln in enumerate(bb_lines):
        if _regex_decref.match(ln) is not None:
            last_decref_pos = pos + 1
    last_pos = max(last_incref_pos, last_decref_pos)
    decrefs = []
    head = []
    for ln in bb_lines[:last_pos]:
        if _regex_decref.match(ln) is not None:
            decrefs.append(ln)
        else:
            head.append(ln)
    return head + decrefs + bb_lines[last_pos:]