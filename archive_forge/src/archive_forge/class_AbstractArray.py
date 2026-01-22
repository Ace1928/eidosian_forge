from __future__ import annotations
import warnings
from collections.abc import Hashable, Iterable, Iterator, Mapping
from contextlib import suppress
from html import escape
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Callable, TypeVar, Union, overload
import numpy as np
import pandas as pd
from xarray.core import dtypes, duck_array_ops, formatting, formatting_html, ops
from xarray.core.indexing import BasicIndexer, ExplicitlyIndexed
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.utils import (
from xarray.namedarray.core import _raise_if_any_duplicate_dimensions
from xarray.namedarray.parallelcompat import get_chunked_array_type, guess_chunkmanager
from xarray.namedarray.pycompat import is_chunked_array
class AbstractArray:
    """Shared base class for DataArray and Variable."""
    __slots__ = ()

    def __bool__(self: Any) -> bool:
        return bool(self.values)

    def __float__(self: Any) -> float:
        return float(self.values)

    def __int__(self: Any) -> int:
        return int(self.values)

    def __complex__(self: Any) -> complex:
        return complex(self.values)

    def __array__(self: Any, dtype: DTypeLike | None=None) -> np.ndarray:
        return np.asarray(self.values, dtype=dtype)

    def __repr__(self) -> str:
        return formatting.array_repr(self)

    def _repr_html_(self):
        if OPTIONS['display_style'] == 'text':
            return f'<pre>{escape(repr(self))}</pre>'
        return formatting_html.array_repr(self)

    def __format__(self: Any, format_spec: str='') -> str:
        if format_spec != '':
            if self.shape == ():
                return self.data.__format__(format_spec)
            else:
                raise NotImplementedError(f'Using format_spec is only supported when shape is (). Got shape = {self.shape}.')
        else:
            return self.__repr__()

    def _iter(self: Any) -> Iterator[Any]:
        for n in range(len(self)):
            yield self[n]

    def __iter__(self: Any) -> Iterator[Any]:
        if self.ndim == 0:
            raise TypeError('iteration over a 0-d array')
        return self._iter()

    @overload
    def get_axis_num(self, dim: Iterable[Hashable]) -> tuple[int, ...]:
        ...

    @overload
    def get_axis_num(self, dim: Hashable) -> int:
        ...

    def get_axis_num(self, dim: Hashable | Iterable[Hashable]) -> int | tuple[int, ...]:
        """Return axis number(s) corresponding to dimension(s) in this array.

        Parameters
        ----------
        dim : str or iterable of str
            Dimension name(s) for which to lookup axes.

        Returns
        -------
        int or tuple of int
            Axis number or numbers corresponding to the given dimensions.
        """
        if not isinstance(dim, str) and isinstance(dim, Iterable):
            return tuple((self._get_axis_num(d) for d in dim))
        else:
            return self._get_axis_num(dim)

    def _get_axis_num(self: Any, dim: Hashable) -> int:
        _raise_if_any_duplicate_dimensions(self.dims)
        try:
            return self.dims.index(dim)
        except ValueError:
            raise ValueError(f'{dim!r} not found in array dimensions {self.dims!r}')

    @property
    def sizes(self: Any) -> Mapping[Hashable, int]:
        """Ordered mapping from dimension names to lengths.

        Immutable.

        See Also
        --------
        Dataset.sizes
        """
        return Frozen(dict(zip(self.dims, self.shape)))