from __future__ import annotations
from collections.abc import Iterable
from typing import (
import numpy as np
from pandas.util._decorators import (
from pandas.core.dtypes.common import (
@cache_readonly
def _positional_selector(self) -> GroupByPositionalSelector:
    """
        Return positional selection for each group.

        ``groupby._positional_selector[i:j]`` is similar to
        ``groupby.apply(lambda x: x.iloc[i:j])``
        but much faster and preserves the original index and order.

        ``_positional_selector[]`` is compatible with and extends :meth:`~GroupBy.head`
        and :meth:`~GroupBy.tail`. For example:

        - ``head(5)``
        - ``_positional_selector[5:-5]``
        - ``tail(5)``

        together return all the rows.

        Allowed inputs for the index are:

        - An integer valued iterable, e.g. ``range(2, 4)``.
        - A comma separated list of integers and slices, e.g. ``5``, ``2, 4``, ``2:4``.

        The output format is the same as :meth:`~GroupBy.head` and
        :meth:`~GroupBy.tail`, namely
        a subset of the ``DataFrame`` or ``Series`` with the index and order preserved.

        Returns
        -------
        Series
            The filtered subset of the original Series.
        DataFrame
            The filtered subset of the original DataFrame.

        See Also
        --------
        DataFrame.iloc : Purely integer-location based indexing for selection by
            position.
        GroupBy.head : Return first n rows of each group.
        GroupBy.tail : Return last n rows of each group.
        GroupBy.nth : Take the nth row from each group if n is an int, or a
            subset of rows, if n is a list of ints.

        Notes
        -----
        - The slice step cannot be negative.
        - If the index specification results in overlaps, the item is not duplicated.
        - If the index specification changes the order of items, then
          they are returned in their original order.
          By contrast, ``DataFrame.iloc`` can change the row order.
        - ``groupby()`` parameters such as as_index and dropna are ignored.

        The differences between ``_positional_selector[]`` and :meth:`~GroupBy.nth`
        with ``as_index=False`` are:

        - Input to ``_positional_selector`` can include
          one or more slices whereas ``nth``
          just handles an integer or a list of integers.
        - ``_positional_selector`` can  accept a slice relative to the
          last row of each group.
        - ``_positional_selector`` does not have an equivalent to the
          ``nth()`` ``dropna`` parameter.

        Examples
        --------
        >>> df = pd.DataFrame([["a", 1], ["a", 2], ["a", 3], ["b", 4], ["b", 5]],
        ...                   columns=["A", "B"])
        >>> df.groupby("A")._positional_selector[1:2]
           A  B
        1  a  2
        4  b  5

        >>> df.groupby("A")._positional_selector[1, -1]
           A  B
        1  a  2
        2  a  3
        4  b  5
        """
    if TYPE_CHECKING:
        groupby_self = cast(groupby.GroupBy, self)
    else:
        groupby_self = self
    return GroupByPositionalSelector(groupby_self)