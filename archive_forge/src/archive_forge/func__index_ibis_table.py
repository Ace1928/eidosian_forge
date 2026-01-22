import sys
from collections.abc import Iterable
from functools import lru_cache
import numpy as np
from packaging.version import Version
from .. import util
from ..element import Element
from ..ndmapping import NdMapping, item_check, sorted_context
from . import pandas
from .interface import Interface
from .util import cached
@classmethod
def _index_ibis_table(cls, data):
    import ibis
    if not cls.has_rowid():
        raise ValueError('iloc expressions are not supported for ibis version %s.' % ibis.__version__)
    if 'hv_row_id__' in data.columns:
        return data
    if ibis4():
        return data.mutate(hv_row_id__=ibis.row_number())
    elif cls.is_rowid_zero_indexed(data):
        return data.mutate(hv_row_id__=data.rowid())
    else:
        return data.mutate(hv_row_id__=data.rowid() - 1)