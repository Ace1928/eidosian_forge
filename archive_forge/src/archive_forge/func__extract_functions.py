import re
from collections import defaultdict, deque
from llvmlite import binding as ll
from numba.core import cgutils
def _extract_functions(module):
    cur = []
    for line in str(module).splitlines():
        if line.startswith('define'):
            assert not cur
            cur.append(line)
        elif line.startswith('}'):
            assert cur
            cur.append(line)
            yield (True, cur)
            cur = []
        elif cur:
            cur.append(line)
        else:
            yield (False, [line])