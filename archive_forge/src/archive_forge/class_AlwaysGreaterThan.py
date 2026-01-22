from __future__ import annotations
import functools
from typing import Any
import numpy as np
from xarray.core import utils
@functools.total_ordering
class AlwaysGreaterThan:

    def __gt__(self, other):
        return True

    def __eq__(self, other):
        return isinstance(other, type(self))