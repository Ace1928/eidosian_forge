import dis
from contextlib import contextmanager
import builtins
import operator
from typing import Iterator
from functools import reduce
from numba.core import (
from numba.core.utils import (
from .rvsdg.bc2rvsdg import (
from .rvsdg.regionpasses import RegionVisitor
def _get_temp_label(self) -> int:
    num = len(self._label_map)
    assert num not in self._label_map
    self._label_map[f'annoy.{num}'] = num
    return num