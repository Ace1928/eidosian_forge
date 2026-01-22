import sys, os
import types
from . import model
from .error import VerificationError
def _loading_gen_struct(self, tp, name, module):
    self._loading_struct_or_union(tp, 'struct', name, module)