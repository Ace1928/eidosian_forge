import sys
from . import model
from .error import VerificationError
from . import _imp_emulation as imp
def _loaded_cpy_enum(self, tp, name, module, library):
    for enumerator, enumvalue in zip(tp.enumerators, tp.enumvalues):
        setattr(library, enumerator, enumvalue)