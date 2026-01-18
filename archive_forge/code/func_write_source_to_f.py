import sys, os
import types
from . import model
from .error import VerificationError
def write_source_to_f(self):
    prnt = self._prnt
    prnt(cffimod_header)
    prnt(self.verifier.preamble)
    self._generate('decl')
    if sys.platform == 'win32':
        if sys.version_info >= (3,):
            prefix = 'PyInit_'
        else:
            prefix = 'init'
        modname = self.verifier.get_module_name()
        prnt('void %s%s(void) { }\n' % (prefix, modname))