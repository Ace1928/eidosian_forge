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
def __get_function_pointer(self, ftype, fname, sig=None):
    from numba.experimental.function_type import lower_get_wrapper_address
    llty = self.context.get_value_type(ftype)
    fstruct = self.loadvar(fname)
    addr = self.builder.extract_value(fstruct, 0, name='addr_of_%s' % fname)
    fptr = cgutils.alloca_once(self.builder, llty, name='fptr_of_%s' % fname)
    with self.builder.if_else(cgutils.is_null(self.builder, addr), likely=False) as (then, orelse):
        with then:
            self.init_pyapi()
            gil_state = self.pyapi.gil_ensure()
            pyaddr = self.builder.extract_value(fstruct, 1, name='pyaddr_of_%s' % fname)
            addr1 = lower_get_wrapper_address(self.context, self.builder, pyaddr, sig, failure_mode='ignore')
            with self.builder.if_then(cgutils.is_null(self.builder, addr1), likely=False):
                self.return_exception(RuntimeError, exc_args=(f'{ftype} function address is null',), loc=self.loc)
            addr2 = self.pyapi.long_as_voidptr(addr1)
            self.builder.store(self.builder.bitcast(addr2, llty), fptr)
            self.pyapi.decref(addr1)
            self.pyapi.gil_release(gil_state)
        with orelse:
            self.builder.store(self.builder.bitcast(addr, llty), fptr)
    return self.builder.load(fptr)