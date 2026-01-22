from collections import namedtuple
from numba.core import types, ir
from numba.core.typing import signature
@property
def _typemap(self):
    return self._lowerer.fndesc.typemap