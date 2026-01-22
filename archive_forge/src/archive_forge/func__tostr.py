from math import gcd
import re
from typing import Dict, Tuple, List, Sequence, Union
from ase.data import chemical_symbols, atomic_numbers
def _tostr(self, sub1, sub2):
    parts = []
    for tree, n in self._tree:
        s = tree2str(tree, sub1, sub2)
        if s[0] == '(' and s[-1] == ')':
            s = s[1:-1]
        if n > 1:
            s = str(n) + s
        parts.append(s)
    return '+'.join(parts)