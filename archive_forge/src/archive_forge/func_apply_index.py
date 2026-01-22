from __future__ import annotations
from contextlib import contextmanager
import copy
from functools import partial
import operator
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
import pandas as pd
from pandas import (
import pandas.core.common as com
from pandas.core.frame import (
from pandas.core.generic import NDFrame
from pandas.core.shared_docs import _shared_docs
from pandas.io.formats.format import save_to_buffer
from pandas.io.formats.style_render import (
@doc(this='apply', wise='level-wise', alt='map', altwise='elementwise', func='take a Series and return a string array of the same length', input_note='the index as a Series, if an Index, or a level of a MultiIndex', output_note='an identically sized array of CSS styles as strings', var='s', ret='np.where(s == "B", "background-color: yellow;", "")', ret2='["background-color: yellow;" if "x" in v else "" for v in s]')
def apply_index(self, func: Callable, axis: AxisInt | str=0, level: Level | list[Level] | None=None, **kwargs) -> Styler:
    """
        Apply a CSS-styling function to the index or column headers, {wise}.

        Updates the HTML representation with the result.

        .. versionadded:: 1.4.0

        .. versionadded:: 2.1.0
           Styler.applymap_index was deprecated and renamed to Styler.map_index.

        Parameters
        ----------
        func : function
            ``func`` should {func}.
        axis : {{0, 1, "index", "columns"}}
            The headers over which to apply the function.
        level : int, str, list, optional
            If index is MultiIndex the level(s) over which to apply the function.
        **kwargs : dict
            Pass along to ``func``.

        Returns
        -------
        Styler

        See Also
        --------
        Styler.{alt}_index: Apply a CSS-styling function to headers {altwise}.
        Styler.apply: Apply a CSS-styling function column-wise, row-wise, or table-wise.
        Styler.map: Apply a CSS-styling function elementwise.

        Notes
        -----
        Each input to ``func`` will be {input_note}. The output of ``func`` should be
        {output_note}, in the format 'attribute: value; attribute2: value2; ...'
        or, if nothing is to be applied to that element, an empty string or ``None``.

        Examples
        --------
        Basic usage to conditionally highlight values in the index.

        >>> df = pd.DataFrame([[1,2], [3,4]], index=["A", "B"])
        >>> def color_b(s):
        ...     return {ret}
        >>> df.style.{this}_index(color_b)  # doctest: +SKIP

        .. figure:: ../../_static/style/appmaphead1.png

        Selectively applying to specific levels of MultiIndex columns.

        >>> midx = pd.MultiIndex.from_product([['ix', 'jy'], [0, 1], ['x3', 'z4']])
        >>> df = pd.DataFrame([np.arange(8)], columns=midx)
        >>> def highlight_x({var}):
        ...     return {ret2}
        >>> df.style.{this}_index(highlight_x, axis="columns", level=[0, 2])
        ...  # doctest: +SKIP

        .. figure:: ../../_static/style/appmaphead2.png
        """
    self._todo.append((lambda instance: getattr(instance, '_apply_index'), (func, axis, level, 'apply'), kwargs))
    return self