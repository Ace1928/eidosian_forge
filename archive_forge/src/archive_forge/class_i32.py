from typing import Any
import sys
from typing import _type_check  # type: ignore
class i32(metaclass=_NativeIntMeta):

    def __new__(cls, x=0, base=_sentinel):
        if base is not _sentinel:
            return int(x, base)
        return int(x)