import sys
from . import model
from .error import VerificationError
from . import _imp_emulation as imp
def _loaded_cpy_function(self, tp, name, module, library):
    if tp.ellipsis:
        return
    func = getattr(module, name)
    setattr(library, name, func)
    self._types_of_builtin_functions[func] = tp