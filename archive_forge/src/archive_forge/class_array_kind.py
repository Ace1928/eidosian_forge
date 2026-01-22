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
class array_kind:

    @staticmethod
    def discrete(arr):
        """
        Return True if array is discrete

        Parameters
        ----------
        arr : numpy.array
            Must have a dtype

        Returns
        -------
        out : bool
            Whether array `arr` is discrete
        """
        return arr.dtype.kind in 'ObUS'

    @staticmethod
    def continuous(arr):
        """
        Return True if array is continuous

        Parameters
        ----------
        arr : numpy.array | pandas.series
            Must have a dtype

        Returns
        -------
        out : bool
            Whether array `arr` is continuous
        """
        return arr.dtype.kind in 'ifuc'

    @staticmethod
    def datetime(arr):
        return arr.dtype.kind == 'M'

    @staticmethod
    def timedelta(arr):
        return arr.dtype.kind == 'm'

    @staticmethod
    def ordinal(arr):
        """
        Return True if array is an ordered categorical

        Parameters
        ----------
        arr : numpy.array
            Must have a dtype

        Returns
        -------
        out : bool
            Whether array `arr` is an ordered categorical
        """
        if isinstance(arr.dtype, pd.CategoricalDtype):
            return arr.cat.ordered
        return False

    @staticmethod
    def categorical(arr):
        """
        Return True if array is a categorical

        Parameters
        ----------
        arr : list-like
            List

        Returns
        -------
        bool
            Whether array `arr` is a categorical
        """
        if not hasattr(arr, 'dtype'):
            return False
        return isinstance(arr.dtype, pd.CategoricalDtype)