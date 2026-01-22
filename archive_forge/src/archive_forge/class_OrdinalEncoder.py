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
class OrdinalEncoder(Preprocessor):
    """Encode values within columns as ordered integer values.

    :class:`OrdinalEncoder` encodes categorical features as integers that range from
    :math:`0` to :math:`n - 1`, where :math:`n` is the number of categories.

    If you transform a value that isn't in the fitted datset, then the value is encoded
    as ``float("nan")``.

    Columns must contain either hashable values or lists of hashable values. Also, you
    can't have both scalars and lists in the same column.

    Examples:
        Use :class:`OrdinalEncoder` to encode categorical features as integers.

        >>> import pandas as pd
        >>> import ray
        >>> from ray.data.preprocessors import OrdinalEncoder
        >>> df = pd.DataFrame({
        ...     "sex": ["male", "female", "male", "female"],
        ...     "level": ["L4", "L5", "L3", "L4"],
        ... })
        >>> ds = ray.data.from_pandas(df)  # doctest: +SKIP
        >>> encoder = OrdinalEncoder(columns=["sex", "level"])
        >>> encoder.fit_transform(ds).to_pandas()  # doctest: +SKIP
           sex  level
        0    1      1
        1    0      2
        2    1      0
        3    0      1

        If you transform a value not present in the original dataset, then the value
        is encoded as ``float("nan")``.

        >>> df = pd.DataFrame({"sex": ["female"], "level": ["L6"]})
        >>> ds = ray.data.from_pandas(df)  # doctest: +SKIP
        >>> encoder.transform(ds).to_pandas()  # doctest: +SKIP
           sex  level
        0    0    NaN

        :class:`OrdinalEncoder` can also encode categories in a list.

        >>> df = pd.DataFrame({
        ...     "name": ["Shaolin Soccer", "Moana", "The Smartest Guys in the Room"],
        ...     "genre": [
        ...         ["comedy", "action", "sports"],
        ...         ["animation", "comedy",  "action"],
        ...         ["documentary"],
        ...     ],
        ... })
        >>> ds = ray.data.from_pandas(df)  # doctest: +SKIP
        >>> encoder = OrdinalEncoder(columns=["genre"])
        >>> encoder.fit_transform(ds).to_pandas()  # doctest: +SKIP
                                    name      genre
        0                 Shaolin Soccer  [2, 0, 4]
        1                          Moana  [1, 2, 0]
        2  The Smartest Guys in the Room        [3]

    Args:
        columns: The columns to separately encode.
        encode_lists: If ``True``, encode list elements.  If ``False``, encode
            whole lists (i.e., replace each list with an integer). ``True``
            by default.

    .. seealso::

        :class:`OneHotEncoder`
            Another preprocessor that encodes categorical data.
    """

    def __init__(self, columns: List[str], *, encode_lists: bool=True):
        self.columns = columns
        self.encode_lists = encode_lists

    def _fit(self, dataset: Dataset) -> Preprocessor:
        self.stats_ = _get_unique_value_indices(dataset, self.columns, encode_lists=self.encode_lists)
        return self

    def _transform_pandas(self, df: pd.DataFrame):
        _validate_df(df, *self.columns)

        def encode_list(element: list, *, name: str):
            return [self.stats_[f'unique_values({name})'].get(x) for x in element]

        def column_ordinal_encoder(s: pd.Series):
            if _is_series_composed_of_lists(s):
                if self.encode_lists:
                    return s.map(partial(encode_list, name=s.name))

                def list_as_category(element):
                    element = tuple(element)
                    return self.stats_[f'unique_values({s.name})'].get(element)
                return s.apply(list_as_category)
            s_values = self.stats_[f'unique_values({s.name})']
            return s.map(s_values)
        df[self.columns] = df[self.columns].apply(column_ordinal_encoder)
        return df

    def __repr__(self):
        return f'{self.__class__.__name__}(columns={self.columns!r}, encode_lists={self.encode_lists!r})'