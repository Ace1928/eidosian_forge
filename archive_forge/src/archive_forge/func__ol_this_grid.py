from numba.core import types
from numba.core.extending import overload, overload_method
from numba.core.typing import signature
from numba.cuda import nvvmutils
from numba.cuda.extending import intrinsic
from numba.cuda.types import grid_group, GridGroup as GridGroupClass
@overload(this_grid, target='cuda')
def _ol_this_grid():

    def impl():
        return _this_grid()
    return impl