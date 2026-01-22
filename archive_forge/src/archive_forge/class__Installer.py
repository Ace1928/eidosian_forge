import sys
from llvmlite import ir
import llvmlite.binding as ll
from numba.core import utils, intrinsics
from numba import _helperlib
class _Installer(object):
    _installed = False

    def install(self, context):
        """
        Install the functions into LLVM.  This only needs to be done once,
        as the mappings are persistent during the process lifetime.
        """
        if not self._installed:
            self._do_install(context)
            self._installed = True