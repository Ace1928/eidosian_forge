import sys, os
import types
from . import model
from .error import VerificationError
def _loaded_gen_anonymous(self, tp, name, module, **kwds):
    if isinstance(tp, model.EnumType):
        self._loaded_gen_enum(tp, name, module, **kwds)
    else:
        self._loaded_struct_or_union(tp)