from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar, cast
import numpy as np
from pandas import DataFrame
from seaborn._core.scales import Scale
from seaborn._core.groupby import GroupBy
from seaborn._stats.base import Stat
from seaborn.utils import _version_predates
@dataclass
class Perc(Stat):
    """
    Replace observations with percentile values.

    Parameters
    ----------
    k : list of numbers or int
        If a list of numbers, this gives the percentiles (in [0, 100]) to compute.
        If an integer, compute `k` evenly-spaced percentiles between 0 and 100.
        For example, `k=5` computes the 0, 25, 50, 75, and 100th percentiles.
    method : str
        Method for interpolating percentiles between observed datapoints.
        See :func:`numpy.percentile` for valid options and more information.

    Examples
    --------
    .. include:: ../docstrings/objects.Perc.rst

    """
    k: int | list[float] = 5
    method: str = 'linear'
    group_by_orient: ClassVar[bool] = True

    def _percentile(self, data: DataFrame, var: str) -> DataFrame:
        k = list(np.linspace(0, 100, self.k)) if isinstance(self.k, int) else self.k
        method = cast(_MethodKind, self.method)
        values = data[var].dropna()
        if _version_predates(np, '1.22'):
            res = np.percentile(values, k, interpolation=method)
        else:
            res = np.percentile(data[var].dropna(), k, method=method)
        return DataFrame({var: res, 'percentile': k})

    def __call__(self, data: DataFrame, groupby: GroupBy, orient: str, scales: dict[str, Scale]) -> DataFrame:
        var = {'x': 'y', 'y': 'x'}[orient]
        return groupby.apply(data, self._percentile, var)