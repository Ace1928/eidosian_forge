from __future__ import annotations
from contextlib import contextmanager
import operator
import numba
from numba import types
from numba.core import cgutils
from numba.core.datamodel import models
from numba.core.extending import (
from numba.core.imputils import impl_ret_borrowed
import numpy as np
from pandas._libs import lib
from pandas.core.indexes.base import Index
from pandas.core.indexing import _iLocIndexer
from pandas.core.internals import SingleBlockManager
from pandas.core.series import Series
class IlocType(types.Type):

    def __init__(self, obj_type) -> None:
        self.obj_type = obj_type
        name = f'iLocIndexer({obj_type})'
        super().__init__(name=name)

    @property
    def key(self):
        return self.obj_type