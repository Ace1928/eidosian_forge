from types import (
from functorch._C import dim as _C
def _py_wrap_method(orig, __torch_function__):

    def impl(*args, **kwargs):
        return __torch_function__(orig, None, args, kwargs)
    return impl