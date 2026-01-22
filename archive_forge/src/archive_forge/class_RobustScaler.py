from typing import List, Tuple
import numpy as np
import pandas as pd
from ray.data import Dataset
from ray.data.aggregate import AbsMax, Max, Mean, Min, Std
from ray.data.preprocessor import Preprocessor
from ray.util.annotations import PublicAPI
@PublicAPI(stability='alpha')
class RobustScaler(Preprocessor):
    """Scale and translate each column using quantiles.

    The general formula is given by

    .. math::
        x' = \\frac{x - \\mu_{1/2}}{\\mu_h - \\mu_l}

    where :math:`x` is the column, :math:`x'` is the transformed column,
    :math:`\\mu_{1/2}` is the column median. :math:`\\mu_{h}` and :math:`\\mu_{l}` are the
    high and low quantiles, respectively. By default, :math:`\\mu_{h}` is the third
    quartile and :math:`\\mu_{l}` is the first quartile.

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
    """

    def __init__(self, columns: List[str], quantile_range: Tuple[float, float]=(0.25, 0.75)):
        self.columns = columns
        self.quantile_range = quantile_range

    def _fit(self, dataset: Dataset) -> Preprocessor:
        low = self.quantile_range[0]
        med = 0.5
        high = self.quantile_range[1]
        num_records = dataset.count()
        max_index = num_records - 1
        split_indices = [int(percentile * max_index) for percentile in (low, med, high)]
        self.stats_ = {}
        for col in self.columns:
            filtered_dataset = dataset.map_batches(lambda df: df[[col]], batch_format='pandas')
            sorted_dataset = filtered_dataset.sort(col)
            _, low, med, high = sorted_dataset.split_at_indices(split_indices)

            def _get_first_value(ds: Dataset, c: str):
                return ds.take(1)[0][c]
            low_val = _get_first_value(low, col)
            med_val = _get_first_value(med, col)
            high_val = _get_first_value(high, col)
            self.stats_[f'low_quantile({col})'] = low_val
            self.stats_[f'median({col})'] = med_val
            self.stats_[f'high_quantile({col})'] = high_val
        return self

    def _transform_pandas(self, df: pd.DataFrame):

        def column_robust_scaler(s: pd.Series):
            s_low_q = self.stats_[f'low_quantile({s.name})']
            s_median = self.stats_[f'median({s.name})']
            s_high_q = self.stats_[f'high_quantile({s.name})']
            diff = s_high_q - s_low_q
            if diff == 0:
                return np.zeros_like(s)
            return (s - s_median) / diff
        df.loc[:, self.columns] = df.loc[:, self.columns].transform(column_robust_scaler)
        return df

    def __repr__(self):
        return f'{self.__class__.__name__}(columns={self.columns!r}, quantile_range={self.quantile_range!r})'