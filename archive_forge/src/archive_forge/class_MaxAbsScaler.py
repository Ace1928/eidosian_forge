from typing import List, Tuple
import numpy as np
import pandas as pd
from ray.data import Dataset
from ray.data.aggregate import AbsMax, Max, Mean, Min, Std
from ray.data.preprocessor import Preprocessor
from ray.util.annotations import PublicAPI
@PublicAPI(stability='alpha')
class MaxAbsScaler(Preprocessor):
    """Scale each column by its absolute max value.

    The general formula is given by

    .. math::

        x' = \\frac{x}{\\max{\\vert x \\vert}}

    where :math:`x` is the column and :math:`x'` is the transformed column. If
    :math:`\\max{\\vert x \\vert} = 0` (i.e., the column contains all zeros), then the
    column is unmodified.

    .. tip::
        This is the recommended way to scale sparse data. If you data isn't sparse,
        you can use :class:`MinMaxScaler` or :class:`StandardScaler` instead.

    Examples:
        >>> import pandas as pd
        >>> import ray
        >>> from ray.data.preprocessors import MaxAbsScaler
        >>>
        >>> df = pd.DataFrame({"X1": [-6, 3], "X2": [2, -4], "X3": [0, 0]})   # noqa: E501
        >>> ds = ray.data.from_pandas(df)  # doctest: +SKIP
        >>> ds.to_pandas()  # doctest: +SKIP
           X1  X2  X3
        0  -6   2   0
        1   3  -4   0

        Columns are scaled separately.

        >>> preprocessor = MaxAbsScaler(columns=["X1", "X2"])
        >>> preprocessor.fit_transform(ds).to_pandas()  # doctest: +SKIP
            X1   X2  X3
        0 -1.0  0.5   0
        1  0.5 -1.0   0

        Zero-valued columns aren't scaled.

        >>> preprocessor = MaxAbsScaler(columns=["X3"])
        >>> preprocessor.fit_transform(ds).to_pandas()  # doctest: +SKIP
           X1  X2   X3
        0  -6   2  0.0
        1   3  -4  0.0

    Args:
        columns: The columns to separately scale.
    """

    def __init__(self, columns: List[str]):
        self.columns = columns

    def _fit(self, dataset: Dataset) -> Preprocessor:
        aggregates = [AbsMax(col) for col in self.columns]
        self.stats_ = dataset.aggregate(*aggregates)
        return self

    def _transform_pandas(self, df: pd.DataFrame):

        def column_abs_max_scaler(s: pd.Series):
            s_abs_max = self.stats_[f'abs_max({s.name})']
            if s_abs_max == 0:
                s_abs_max = 1
            return s / s_abs_max
        df.loc[:, self.columns] = df.loc[:, self.columns].transform(column_abs_max_scaler)
        return df

    def __repr__(self):
        return f'{self.__class__.__name__}(columns={self.columns!r})'