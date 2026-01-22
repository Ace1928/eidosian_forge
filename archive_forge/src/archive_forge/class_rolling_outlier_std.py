import numpy as np
import pandas as pd
import param
from packaging.version import Version
from ..core import Element, Operation
from ..core.data import PandasInterface
from ..core.util import _PANDAS_FUNC_LOOKUP, pandas_version
from ..element import Scatter
class rolling_outlier_std(Operation, RollingBase):
    """
    Detect outliers using the standard deviation within a rolling window.

    Outliers are the array elements outside `sigma` standard deviations from
    the smoothed trend line, as calculated from the trend line residuals.

    The rolling window is controlled by parameters shared with the
    `rolling` operation via the base class RollingBase, to make it
    simpler to use the same settings for both.
    """
    sigma = param.Number(default=2.0, doc='\n        Minimum sigma before a value is considered an outlier.')

    def _process_layer(self, element, key=None):
        ys = element.dimension_values(1)
        avg = pd.Series(ys).rolling(**self._roll_kwargs()).mean()
        residual = ys - avg
        std = pd.Series(residual).rolling(**self._roll_kwargs()).std()
        with np.errstate(invalid='ignore'):
            outliers = (np.abs(residual) > std * self.p.sigma).values
        return element[outliers].clone(new_type=Scatter)

    def _process(self, element, key=None):
        return element.map(self._process_layer, Element)