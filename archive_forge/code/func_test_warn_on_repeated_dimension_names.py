from __future__ import annotations
import copy
import warnings
from abc import abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Generic, cast, overload
import numpy as np
import pytest
from xarray.core.indexing import ExplicitlyIndexed
from xarray.namedarray._typing import (
from xarray.namedarray.core import NamedArray, from_array
def test_warn_on_repeated_dimension_names(self) -> None:
    with pytest.warns(UserWarning, match='Duplicate dimension names'):
        NamedArray(('x', 'x'), np.arange(4).reshape(2, 2))