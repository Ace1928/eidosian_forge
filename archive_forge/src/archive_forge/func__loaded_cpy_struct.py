import sys
from . import model
from .error import VerificationError
from . import _imp_emulation as imp
def _loaded_cpy_struct(self, tp, name, module, **kwds):
    self._loaded_struct_or_union(tp)