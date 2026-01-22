from __future__ import annotations
import inspect
import itertools
import warnings
from collections import defaultdict
from collections.abc import Iterable, Sequence
from contextlib import suppress
from copy import deepcopy
from dataclasses import field
from typing import TYPE_CHECKING, cast, overload
from warnings import warn
import numpy as np
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy
from ..exceptions import PlotnineError, PlotnineWarning
from ..mapping import aes
def _margins(vars: tuple[Sequence[str], Sequence[str]], margins: bool | Sequence[str]=True):
    """
    Figure out margining variables.

    Given the variables that form the rows and
    columns, and a set of desired margins, works
    out which ones are possible. Variables that
    can't be margined over are dropped silently.

    Parameters
    ----------
    vars : list
        variable names for rows and columns
    margins : bool | list
        If true, margins over all vars, otherwise
        only those listed

    Return
    ------
    out : list
        All the margins to create.
    """
    if margins is False:
        return []

    def fn(_vars):
        """The margin variables for a given row or column"""
        dim_margins = [[]]
        for i, u in enumerate(_vars):
            if margins is True or u in margins:
                lst = [u] + list(_vars[i + 1:])
                dim_margins.append(lst)
        return dim_margins
    row_margins = fn(vars[0])
    col_margins = fn(vars[1])
    lst = list(itertools.product(col_margins, row_margins))
    pretty = []
    for c, r in lst:
        pretty.append(r + c)
    return pretty