from __future__ import annotations
import textwrap
from typing import (
import warnings
import numpy as np
from pandas._config import using_copy_on_write
from pandas._libs import lib
from pandas._typing import (
from pandas.compat import PYPY
from pandas.compat.numpy import function as nv
from pandas.errors import AbstractMethodError
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import can_hold_element
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core.accessor import DirNamesMixin
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays import ExtensionArray
from pandas.core.construction import (
@final
@cache_readonly
def _obj_with_exclusions(self):
    if isinstance(self.obj, ABCSeries):
        return self.obj
    if self._selection is not None:
        return self.obj._getitem_nocopy(self._selection_list)
    if len(self.exclusions) > 0:
        return self.obj._drop_axis(self.exclusions, axis=1, only_slice=True)
    else:
        return self.obj