from llvmlite import ir
from numba import cuda, types
from numba.core import cgutils
from numba.core.errors import RequireLiteralValue
from numba.core.typing import signature
from numba.core.extending import overload_attribute
from numba.cuda import nvvmutils
from numba.cuda.extending import intrinsic
def _nthreads_for_dim(builder, dim):
    i64 = ir.IntType(64)
    ntid = nvvmutils.call_sreg(builder, f'ntid.{dim}')
    nctaid = nvvmutils.call_sreg(builder, f'nctaid.{dim}')
    return builder.mul(builder.sext(ntid, i64), builder.sext(nctaid, i64))