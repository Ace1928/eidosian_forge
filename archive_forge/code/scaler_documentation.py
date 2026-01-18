from typing import List, Tuple
import numpy as np
import pandas as pd
from ray.data import Dataset
from ray.data.aggregate import AbsMax, Max, Mean, Min, Std
from ray.data.preprocessor import Preprocessor
from ray.util.annotations import PublicAPI
Scale and translate each column using quantiles.

    The general formula is given by

    .. math::
        x' = \frac{x - \mu_{1/2}}{\mu_h - \mu_l}

    where :math:`x` is the column, :math:`x'` is the transformed column,
    :math:`\mu_{1/2}` is the column median. :math:`\mu_{h}` and :math:`\mu_{l}` are the
    high and low quantiles, respectively. By default, :math:`\mu_{h}` is the third
    quartile and :math:`\mu_{l}` is the first quartile.

    .. tip::
        This scaler works well when your data contains many outliers.

    Examples:
        >>> import pandas as pd
        >>> import ray
        >>> from ray.data.preprocessors import RobustScaler
        >>>
        >>> df = pd.DataFrame({
        ...     "X1": [1, 2, 3, 4, 5],
        ...     "X2": [13, 5, 14, 2, 8],
        ...     "X3": [1, 2, 2, 2, 3],
        ... })
        >>> ds = ray.data.from_pandas(df)  # doctest: +SKIP
        >>> ds.to_pandas()  # doctest: +SKIP
           X1  X2  X3
        0   1  13   1
        1   2   5   2
        2   3  14   2
        3   4   2   2
        4   5   8   3

        :class:`RobustScaler` separately scales each column.

        >>> preprocessor = RobustScaler(columns=["X1", "X2"])
        >>> preprocessor.fit_transform(ds).to_pandas()  # doctest: +SKIP
            X1     X2  X3
        0 -1.0  0.625   1
        1 -0.5 -0.375   2
        2  0.0  0.750   2
        3  0.5 -0.750   2
        4  1.0  0.000   3

    Args:
        columns: The columns to separately scale.
        quantile_range: A tuple that defines the lower and upper quantiles. Values
            must be between 0 and 1. Defaults to the 1st and 3rd quartiles:
            ``(0.25, 0.75)``.
    