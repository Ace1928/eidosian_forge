from cupyx.jit import _cuda_types
from cupyx.jit import _cuda_typerules
from cupyx.jit import _internal_types
from cupy_backends.cuda.api import runtime as _runtime
def _include_cub(env):
    if _runtime.is_hip:
        env.generated.add_code('#include <hipcub/hipcub.hpp>')
    elif _runtime.runtimeGetVersion() < 11000:
        env.generated.add_code('#include <cupy/cub/cub/cub.cuh>')
    else:
        env.generated.add_code('#include <cub/cub.cuh>')
    env.generated.backend = 'nvcc'