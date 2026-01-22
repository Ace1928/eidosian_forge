from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from seaborn._stats.base import Stat
@dataclass
class PolyFit(Stat):
    """
    Fit a polynomial of the given order and resample data onto predicted curve.
    """
    order: int = 2
    gridsize: int = 100

    def _fit_predict(self, data):
        x = data['x']
        y = data['y']
        if x.nunique() <= self.order:
            xx = yy = []
        else:
            p = np.polyfit(x, y, self.order)
            xx = np.linspace(x.min(), x.max(), self.gridsize)
            yy = np.polyval(p, xx)
        return pd.DataFrame(dict(x=xx, y=yy))

    def __call__(self, data, groupby, orient, scales):
        return groupby.apply(data.dropna(subset=['x', 'y']), self._fit_predict)