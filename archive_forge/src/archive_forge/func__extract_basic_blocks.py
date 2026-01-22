import re
from collections import defaultdict, deque
from llvmlite import binding as ll
from numba.core import cgutils
def _extract_basic_blocks(func_lines):
    assert func_lines[0].startswith('define')
    assert func_lines[-1].startswith('}')
    yield (False, [func_lines[0]])
    cur = []
    for ln in func_lines[1:-1]:
        m = _regex_bb.match(ln)
        if m is not None:
            yield (True, cur)
            cur = []
            yield (False, [ln])
        elif ln:
            cur.append(ln)
    yield (True, cur)
    yield (False, [func_lines[-1]])