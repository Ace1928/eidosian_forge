from llvmlite.ir import Constant, IRBuilder
import llvmlite.ir
from numba.core import types, config, cgutils
def _simplified_return_type(self):
    """
        The NPM callconv has already converted simplified optional types.
        We can simply use the value type from it.
        """
    restype = self.fndesc.restype
    if isinstance(restype, types.Optional):
        return restype.type
    else:
        return restype