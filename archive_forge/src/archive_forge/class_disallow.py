from __future__ import annotations
import functools
import itertools
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import (
from pandas._typing import (
from pandas.compat._optional import import_optional_dependency
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.missing import (
class disallow:

    def __init__(self, *dtypes: Dtype) -> None:
        super().__init__()
        self.dtypes = tuple((pandas_dtype(dtype).type for dtype in dtypes))

    def check(self, obj) -> bool:
        return hasattr(obj, 'dtype') and issubclass(obj.dtype.type, self.dtypes)

    def __call__(self, f: F) -> F:

        @functools.wraps(f)
        def _f(*args, **kwargs):
            obj_iter = itertools.chain(args, kwargs.values())
            if any((self.check(obj) for obj in obj_iter)):
                f_name = f.__name__.replace('nan', '')
                raise TypeError(f"reduction operation '{f_name}' not allowed for this dtype")
            try:
                return f(*args, **kwargs)
            except ValueError as e:
                if is_object_dtype(args[0]):
                    raise TypeError(e) from e
                raise
        return cast(F, _f)