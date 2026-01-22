from __future__ import annotations
import functools
import itertools
import warnings
from collections.abc import Hashable, Iterable, MutableMapping
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, TypeVar, cast
import numpy as np
from xarray.core.formatting import format_item
from xarray.core.types import HueStyleOptions, T_DataArrayOrSet
from xarray.plot.utils import (
def add_quiverkey(self, u: Hashable, v: Hashable, **kwargs: Any) -> None:
    kwargs = kwargs.copy()
    magnitude = _get_nice_quiver_magnitude(self.data[u], self.data[v])
    units = self.data[u].attrs.get('units', '')
    self.quiverkey = self.axs.flat[-1].quiverkey(self._mappables[-1], X=0.8, Y=0.9, U=magnitude, label=f'{magnitude}\n{units}', labelpos='E', coordinates='figure')