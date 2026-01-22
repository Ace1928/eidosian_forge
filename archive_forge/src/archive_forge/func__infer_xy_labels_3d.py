from __future__ import annotations
import itertools
import textwrap
import warnings
from collections.abc import Hashable, Iterable, Mapping, MutableMapping, Sequence
from datetime import date, datetime
from inspect import getfullargspec
from typing import TYPE_CHECKING, Any, Callable, Literal, overload
import numpy as np
import pandas as pd
from xarray.core.indexes import PandasMultiIndex
from xarray.core.options import OPTIONS
from xarray.core.utils import is_scalar, module_available
from xarray.namedarray.pycompat import DuckArrayModule
def _infer_xy_labels_3d(darray: DataArray | Dataset, x: Hashable | None, y: Hashable | None, rgb: Hashable | None) -> tuple[Hashable, Hashable]:
    """
    Determine x and y labels for showing RGB images.

    Attempts to infer which dimension is RGB/RGBA by size and order of dims.

    """
    assert rgb is None or rgb != x
    assert rgb is None or rgb != y
    assert darray.ndim == 3
    not_none = [a for a in (x, y, rgb) if a is not None]
    if len(set(not_none)) < len(not_none):
        raise ValueError(f'Dimension names must be None or unique strings, but imshow was passed x={x!r}, y={y!r}, and rgb={rgb!r}.')
    for label in not_none:
        if label not in darray.dims:
            raise ValueError(f'{label!r} is not a dimension')
    could_be_color = [label for label in darray.dims if darray[label].size in (3, 4) and label not in (x, y)]
    if rgb is None and (not could_be_color):
        raise ValueError('A 3-dimensional array was passed to imshow(), but there is no dimension that could be color.  At least one dimension must be of size 3 (RGB) or 4 (RGBA), and not given as x or y.')
    if rgb is None and len(could_be_color) == 1:
        rgb = could_be_color[0]
    if rgb is not None and darray[rgb].size not in (3, 4):
        raise ValueError(f'Cannot interpret dim {rgb!r} of size {darray[rgb].size} as RGB or RGBA.')
    if rgb is None:
        assert len(could_be_color) in (2, 3)
        rgb = could_be_color[-1]
        warnings.warn(f'Several dimensions of this array could be colors.  Xarray will use the last possible dimension ({rgb!r}) to match matplotlib.pyplot.imshow.  You can pass names of x, y, and/or rgb dimensions to override this guess.')
    assert rgb is not None
    return _infer_xy_labels(darray.isel({rgb: 0}), x, y)