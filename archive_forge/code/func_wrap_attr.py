from types import (
from functorch._C import dim as _C
def wrap_attr(orig):
    return property(wrap_method(orig.__get__, __torch_function__))