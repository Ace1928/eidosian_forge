from .sage_helper import _within_sage
from .pari import *
import re
def add_number_method(name, include_precision=True):
    method = getattr(Gen, name)
    if include_precision:
        setattr(Number, name, lambda self: self.parent()(method(self.gen, precision=self._precision)))
    else:
        setattr(Number, name, lambda self: self.parent()(method(self.gen)))