import copy
import itertools
from pyomo.common import DeveloperError
from pyomo.common.collections import Sequence
from pyomo.core.base.enums import SortComponents
from pyomo.core.base.global_set import UnindexedComponent_index
@classmethod
def _getitem_args_to_str(cls, args):
    for i, v in enumerate(args):
        if v is Ellipsis:
            args[i] = '...'
        elif type(v) is slice:
            args[i] = (repr(v.start) if v.start is not None else '') + ':' + (repr(v.stop) if v.stop is not None else '') + (':%r' % v.step if v.step is not None else '')
        else:
            args[i] = repr(v)
    return '[' + ', '.join(args) + ']'