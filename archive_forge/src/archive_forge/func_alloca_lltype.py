from collections import namedtuple, defaultdict
import operator
import warnings
from functools import partial
import llvmlite.ir
from llvmlite.ir import Constant, IRBuilder
from numba.core import (typing, utils, types, ir, debuginfo, funcdesc,
from numba.core.errors import (LoweringError, new_error_context, TypingError,
from numba.core.funcdesc import default_mangler
from numba.core.environment import Environment
from numba.core.analysis import compute_use_defs, must_use_alloca
from numba.misc.firstlinefinder import get_func_body_first_lineno
def alloca_lltype(self, name, lltype, datamodel=None):
    is_uservar = not name.startswith('$')
    aptr = cgutils.alloca_once(self.builder, lltype, name=name, zfill=False)
    if is_uservar:
        if name not in self.func_ir.arg_names:
            sizeof = self.context.get_abi_sizeof(lltype)
            self.debuginfo.mark_variable(self.builder, aptr, name=name, lltype=lltype, size=sizeof, line=self.loc.line, datamodel=datamodel)
    return aptr