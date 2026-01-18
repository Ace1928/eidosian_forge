from numba.core import types, errors, ir, sigutils, ir_utils
from numba.core.typing.typeof import typeof_impl
from numba.core.transforms import find_region_inout_vars
from numba.core.ir_utils import build_definitions
import numba
@typeof_impl.register(WithContext)
def typeof_contextmanager(val, c):
    return types.ContextManager(val)