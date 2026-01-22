import sys, os
import types
from . import model
from .error import VerificationError
def _generate_gen_struct_decl(self, tp, name):
    assert name == tp.name
    self._generate_struct_or_union_decl(tp, 'struct', name)