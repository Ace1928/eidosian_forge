from __future__ import annotations
from collections.abc import Hashable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import (
import numpy as np
import pandas as pd
from xarray.core import formatting
from xarray.core.alignment import Aligner
from xarray.core.indexes import (
from xarray.core.merge import merge_coordinates_without_align, merge_coords
from xarray.core.types import DataVars, Self, T_DataArray, T_Xarray
from xarray.core.utils import (
from xarray.core.variable import Variable, as_variable, calculate_dimensions
def _merge_raw(self, other, reflexive):
    """For use with binary arithmetic."""
    if other is None:
        variables = dict(self.variables)
        indexes = dict(self.xindexes)
    else:
        coord_list = [self, other] if not reflexive else [other, self]
        variables, indexes = merge_coordinates_without_align(coord_list)
    return (variables, indexes)