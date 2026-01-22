import sys
from . import model
from .error import VerificationError
from . import _imp_emulation as imp
def _generate_struct_or_union_method(self, tp, prefix, name):
    if tp.fldnames is None:
        return
    layoutfuncname = '_cffi_layout_%s_%s' % (prefix, name)
    self._prnt('  {"%s", %s, METH_NOARGS, NULL},' % (layoutfuncname, layoutfuncname))