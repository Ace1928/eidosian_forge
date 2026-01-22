import sys
from . import model
from .error import VerificationError
from . import _imp_emulation as imp
def _generate_cpy_union_method(self, tp, name):
    self._generate_struct_or_union_method(tp, 'union', name)