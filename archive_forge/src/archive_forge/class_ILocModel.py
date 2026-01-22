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
@register_model(IlocType)
class ILocModel(models.StructModel):

    def __init__(self, dmm, fe_type) -> None:
        members = [('obj', fe_type.obj_type)]
        models.StructModel.__init__(self, dmm, fe_type, members)