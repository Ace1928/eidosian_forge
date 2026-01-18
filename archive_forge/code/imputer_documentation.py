from collections import Counter
from numbers import Number
from typing import Dict, List, Optional, Union
import pandas as pd
from pandas.api.types import is_categorical_dtype
from ray.data import Dataset
from ray.data.aggregate import Mean
from ray.data.preprocessor import Preprocessor
from ray.util.annotations import PublicAPI
Replace missing values with imputed values.

    Examples:
        >>> import pandas as pd
        >>> import ray
        >>> from ray.data.preprocessors import SimpleImputer
        >>> df = pd.DataFrame({"X": [0, None, 3, 3], "Y": [None, "b", "c", "c"]})
        >>> ds = ray.data.from_pandas(df)  # doctest: +SKIP
        >>> ds.to_pandas()  # doctest: +SKIP
             X     Y
        0  0.0  None
        1  NaN     b
        2  3.0     c
        3  3.0     c

        The `"mean"` strategy imputes missing values with the mean of non-missing
        values. This strategy doesn't work with categorical data.

        >>> preprocessor = SimpleImputer(columns=["X"], strategy="mean")
        >>> preprocessor.fit_transform(ds).to_pandas()  # doctest: +SKIP
             X     Y
        0  0.0  None
        1  2.0     b
        2  3.0     c
        3  3.0     c

        The `"most_frequent"` strategy imputes missing values with the most frequent
        value in each column.

        >>> preprocessor = SimpleImputer(columns=["X", "Y"], strategy="most_frequent")
        >>> preprocessor.fit_transform(ds).to_pandas()  # doctest: +SKIP
             X  Y
        0  0.0  c
        1  3.0  b
        2  3.0  c
        3  3.0  c

        The `"constant"` strategy imputes missing values with the value specified by
        `fill_value`.

        >>> preprocessor = SimpleImputer(
        ...     columns=["Y"],
        ...     strategy="constant",
        ...     fill_value="?",
        ... )
        >>> preprocessor.fit_transform(ds).to_pandas()  # doctest: +SKIP
             X  Y
        0  0.0  ?
        1  NaN  b
        2  3.0  c
        3  3.0  c

    Args:
        columns: The columns to apply imputation to.
        strategy: How imputed values are chosen.

            * ``"mean"``: The mean of non-missing values. This strategy only works with numeric columns.
            * ``"most_frequent"``: The most common value.
            * ``"constant"``: The value passed to ``fill_value``.

        fill_value: The value to use when ``strategy`` is ``"constant"``.

    Raises:
        ValueError: if ``strategy`` is not ``"mean"``, ``"most_frequent"``, or
            ``"constant"``.
    