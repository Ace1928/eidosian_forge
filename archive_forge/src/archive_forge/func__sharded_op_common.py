import functools
from torch.distributed._shard.sharded_tensor import (
from torch.distributed._shard.common_op_utils import _basic_validation
def _sharded_op_common(op, early_stop_func, extra_check):
    """
    Inject sharded tensor op registration with common logics executed before
    different behaviors are done on either local shards or a local tensor.

    Example::
        >>> # xdoctest: +SKIP("Undefined variables")
        >>> op = torch.transpose
        >>> @_sharded_op_impl(op)
        >>> @_sharded_op_common(op, early_stop_func, extra_check)
        >>> def sharded_tensor_op(types, args, kwargs, process_group):
        >>>   ...
        >>>
        >>> st = sharded_tensor.rand(32, 16)
        >>> st.transpose(1, 2)
        >>> # This will call '_sharded_op_common'

    Args:
        op: The op to be registered and applied to all shards of the st.
        early_stop_func (Callable, optional): the func for early stop.
            Default: if ``None``, no early stop.
        extra_check (Callable, optional): the func for extra condition check.
            Default: if ``None``, no extra check.

    Return:
        func (Callable): Torch function for which we want to provide a sharded
            implementation (ex: torch.transpose)
    """

    def decorator_sharded_func(wrapped_func):

        @functools.wraps(wrapped_func)
        def wrapper(types, args=(), kwargs=None, pg=None):
            _basic_validation(op, args, kwargs)
            st = args[0]
            if kwargs is None:
                kwargs = {}
            if extra_check:
                extra_check(*args, **kwargs)
            if early_stop_func:
                early_stop = early_stop_func(*args, **kwargs)
                if early_stop:
                    return st
            return wrapped_func(types, args, kwargs, pg)
        return wrapper
    return decorator_sharded_func