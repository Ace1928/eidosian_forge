import codecs
import re
import ply.lex
from pyomo.common.collections import ComponentMap
from pyomo.common.dependencies import pickle
from pyomo.common.deprecation import deprecated
from pyomo.core.base.component_namer import (
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.reference import Reference
def _parse_cuid_v1(self, label):
    """Parse a string (v1 repr format) and yield name, idx pairs

        This attempts to parse a string (nominally returned by
        get_repr()) to generate the sequence of (name, idx) pairs for
        the _cuids data structure.

        """
    cList = label.split('.')
    for c in cList:
        if c[-1] == ']':
            c_info = c[:-1].split('[', 1)
        else:
            c_info = c.split(':', 1)
        if len(c_info) == 1:
            yield (c_info[0], tuple())
        else:
            idx = c_info[1].split(',')
            for i, val in enumerate(idx):
                if val == '*':
                    idx[i] = slice(None)
                elif val[0] == '$':
                    idx[i] = str(val[1:])
                elif val[0] == '#':
                    idx[i] = _int_or_float(val[1:])
                elif val[0] in '"\'' and val[-1] == val[0]:
                    idx[i] = val[1:-1]
                elif _re_number.match(val):
                    idx[i] = _int_or_float(val)
            if len(idx) == 1 and idx[0] == '**':
                yield (c_info[0], (Ellipsis,))
            else:
                yield (c_info[0], tuple(idx))