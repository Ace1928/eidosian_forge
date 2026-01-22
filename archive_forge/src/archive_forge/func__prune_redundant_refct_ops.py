import re
from collections import defaultdict, deque
from llvmlite import binding as ll
from numba.core import cgutils
def _prune_redundant_refct_ops(bb_lines):
    incref_map = defaultdict(deque)
    decref_map = defaultdict(deque)
    to_remove = set()
    for num, incref_var, decref_var in _examine_refct_op(bb_lines):
        assert not (incref_var and decref_var)
        if incref_var:
            if incref_var == 'i8* null':
                to_remove.add(num)
            else:
                incref_map[incref_var].append(num)
        elif decref_var:
            if decref_var == 'i8* null':
                to_remove.add(num)
            else:
                decref_map[decref_var].append(num)
    for var, decops in decref_map.items():
        incops = incref_map[var]
        ct = min(len(incops), len(decops))
        for _ in range(ct):
            to_remove.add(incops.pop())
            to_remove.add(decops.popleft())
    return [ln for num, ln in enumerate(bb_lines) if num not in to_remove]