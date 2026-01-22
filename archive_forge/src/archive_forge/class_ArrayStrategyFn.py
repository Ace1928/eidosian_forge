from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, Union, overload
import hypothesis.extra.numpy as npst
import numpy as np
from hypothesis.errors import InvalidArgument
import xarray as xr
from xarray.core.types import T_DuckArray
class ArrayStrategyFn(Protocol[T_DuckArray]):

    def __call__(self, *, shape: '_ShapeLike', dtype: '_DTypeLikeNested') -> st.SearchStrategy[T_DuckArray]:
        ...