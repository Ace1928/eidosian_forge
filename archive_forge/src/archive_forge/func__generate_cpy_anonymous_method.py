import sys
from . import model
from .error import VerificationError
from . import _imp_emulation as imp
def _generate_cpy_anonymous_method(self, tp, name):
    if not isinstance(tp, model.EnumType):
        self._generate_struct_or_union_method(tp, '', name)