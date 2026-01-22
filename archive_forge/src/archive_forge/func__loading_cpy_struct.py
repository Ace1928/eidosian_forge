import sys
from . import model
from .error import VerificationError
from . import _imp_emulation as imp
def _loading_cpy_struct(self, tp, name, module):
    self._loading_struct_or_union(tp, 'struct', name, module)