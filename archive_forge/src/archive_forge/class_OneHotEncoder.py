from collections import Counter, OrderedDict
from functools import partial
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import pandas.api.types
from ray.air.util.data_batch_conversion import BatchFormat
from ray.data import Dataset
from ray.data.preprocessor import Preprocessor, PreprocessorNotFittedException
from ray.util.annotations import PublicAPI
@PublicAPI(stability='alpha')
class OneHotEncoder(Preprocessor):
    """`One-hot encode <https://en.wikipedia.org/wiki/One-hot#Machine_learning_and_statistics>`_
    categorical data.

    This preprocessor creates a column named ``{column}_{category}``
    for each unique ``{category}`` in ``{column}``. The value of a column is
    1 if the category matches and 0 otherwise.

    If you encode an infrequent category (see ``max_categories``) or a category
    that isn't in the fitted dataset, then the category is encoded as all 0s.

    Columns must contain hashable objects or lists of hashable objects.

    .. note::
        Lists are treated as categories. If you want to encode individual list
        elements, use :class:`MultiHotEncoder`.

    Example:
        >>> import pandas as pd
        >>> import ray
        >>> from ray.data.preprocessors import OneHotEncoder
        >>>
        >>> df = pd.DataFrame({"color": ["red", "green", "red", "red", "blue", "green"]})
        >>> ds = ray.data.from_pandas(df)  # doctest: +SKIP
        >>> encoder = OneHotEncoder(columns=["color"])
        >>> encoder.fit_transform(ds).to_pandas()  # doctest: +SKIP
           color_blue  color_green  color_red
        0           0            0          1
        1           0            1          0
        2           0            0          1
        3           0            0          1
        4           1            0          0
        5           0            1          0

        If you one-hot encode a value that isn't in the fitted dataset, then the
        value is encoded with zeros.

        >>> df = pd.DataFrame({"color": ["yellow"]})
        >>> batch = ray.data.from_pandas(df)  # doctest: +SKIP
        >>> encoder.transform(batch).to_pandas()  # doctest: +SKIP
           color_blue  color_green  color_red
        0           0            0          0

        Likewise, if you one-hot encode an infrequent value, then the value is encoded
        with zeros.

        >>> encoder = OneHotEncoder(columns=["color"], max_categories={"color": 2})
        >>> encoder.fit_transform(ds).to_pandas()  # doctest: +SKIP
           color_red  color_green
        0          1            0
        1          0            1
        2          1            0
        3          1            0
        4          0            0
        5          0            1

    Args:
        columns: The columns to separately encode.
        max_categories: The maximum number of features to create for each column.
            If a value isn't specified for a column, then a feature is created
            for every category in that column.

    .. seealso::

        :class:`MultiHotEncoder`
            If you want to encode individual list elements, use
            :class:`MultiHotEncoder`.

        :class:`OrdinalEncoder`
            If your categories are ordered, you may want to use
            :class:`OrdinalEncoder`.
    """

    def __init__(self, columns: List[str], *, max_categories: Optional[Dict[str, int]]=None):
        self.columns = columns
        self.max_categories = max_categories

    def _fit(self, dataset: Dataset) -> Preprocessor:
        self.stats_ = _get_unique_value_indices(dataset, self.columns, max_categories=self.max_categories, encode_lists=False)
        return self

    def _transform_pandas(self, df: pd.DataFrame):
        _validate_df(df, *self.columns)
        columns_to_drop = set(self.columns)
        for column in self.columns:
            column_values = self.stats_[f'unique_values({column})']
            if _is_series_composed_of_lists(df[column]):
                df[column] = df[column].map(lambda x: tuple(x))
            for column_value in column_values:
                df[f'{column}_{column_value}'] = (df[column] == column_value).astype(int)
        df = df.drop(columns=list(columns_to_drop))
        return df

    def __repr__(self):
        return f'{self.__class__.__name__}(columns={self.columns!r}, max_categories={self.max_categories!r})'