from cupyx.jit import _internal_types
from cupyx.jit import _cuda_types
from cupyx.jit._internal_types import Data as _Data
@_wrap_thrust_func(['thrust/unique.h'])
def unique_by_key(env, exec_policy, keys_first, keys_last, values_first, binary_pred=None):
    """Uniques key-value pairs.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_pointer_type(keys_first)
    _assert_same_type(keys_first, keys_last)
    _assert_pointer_type(values_first)
    args = [exec_policy, keys_first, keys_last, values_first]
    if binary_pred is not None:
        raise NotImplementedError('binary_pred option is not supported')
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::unique_by_key({params})', _cuda_types.Tuple([keys_first.ctype, values_first.ctype]))