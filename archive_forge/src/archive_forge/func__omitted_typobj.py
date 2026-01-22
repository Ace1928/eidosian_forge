import builtins
import operator
import inspect
from functools import cached_property
import llvmlite.ir
from numba.core import types, utils, ir, generators, cgutils
from numba.core.errors import (ForbiddenConstruct, LoweringError,
from numba.core.lowering import BaseLower
@cached_property
def _omitted_typobj(self):
    """Return a `OmittedArg` type instance as a LLVM value suitable for
        testing at runtime.
        """
    from numba.core.dispatcher import OmittedArg
    return self.pyapi.unserialize(self.pyapi.serialize_object(OmittedArg))