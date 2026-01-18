from __future__ import absolute_import
from . import Naming
from . import PyrexTypes
from .Errors import error
import copy
def with_fastcall(self):
    sig = copy.copy(self)
    sig.use_fastcall = True
    return sig