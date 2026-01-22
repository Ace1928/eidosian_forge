from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable
import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame
from seaborn._core.groupby import GroupBy
from seaborn._core.scales import Scale
from seaborn._stats.base import Stat
def _fit_and_evaluate(self, data: DataFrame, orient: str, support: ndarray) -> DataFrame:
    """Transform single group by fitting a KDE and evaluating on a support grid."""
    empty = pd.DataFrame(columns=[orient, 'weight', 'density'], dtype=float)
    if len(data) < 2:
        return empty
    try:
        kde = self._fit(data, orient)
    except np.linalg.LinAlgError:
        return empty
    if self.cumulative:
        s_0 = support[0]
        density = np.array([kde.integrate_box_1d(s_0, s_i) for s_i in support])
    else:
        density = kde(support)
    weight = data['weight'].sum()
    return pd.DataFrame({orient: support, 'weight': weight, 'density': density})