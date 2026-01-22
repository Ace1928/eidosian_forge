import types
from collections import namedtuple
from copy import deepcopy
from weakref import ref as _weakref_ref
def _deepcopier(obj, memo, _id):
    return deepcopy(obj, memo)