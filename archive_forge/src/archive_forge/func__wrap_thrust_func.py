from cupyx.jit import _internal_types
from cupyx.jit import _cuda_types
from cupyx.jit._internal_types import Data as _Data
def _wrap_thrust_func(headers):

    def wrapper(func):

        class FuncWrapper(_internal_types.BuiltinFunc):

            def call(self, env, *args, **kwargs):
                for header in headers:
                    env.generated.add_code(f'#include <{header}>')
                env.generated.add_code('#include <thrust/execution_policy.h>')
                env.generated.add_code('#include <thrust/functional.h>')
                env.generated.backend = 'nvcc'
                data_args = [_Data.init(a, env) for a in args]
                data_kwargs = {k: _Data.init(kwargs[k], env) for k in kwargs}
                return func(env, *data_args, **data_kwargs)
        return FuncWrapper()
    return wrapper