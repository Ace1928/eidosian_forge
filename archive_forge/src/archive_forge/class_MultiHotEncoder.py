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
class MultiHotEncoder(Preprocessor):
    """Multi-hot encode categorical data.

    This preprocessor replaces each list of categories with an :math:`m`-length binary
    list, where :math:`m` is the number of unique categories in the column or the value
    specified in ``max_categories``. The :math:`i\\text{-th}` element of the binary list
    is :math:`1` if category :math:`i` is in the input list and :math:`0` otherwise.

    Columns must contain hashable objects or lists of hashable objects.
    Also, you can't have both types in the same column.

    .. note::
        The logic is similar to scikit-learn's `MultiLabelBinarizer     <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing    .MultiLabelBinarizer.html>`_.

    Examples:
        >>> import pandas as pd
        >>> import ray
        >>> from ray.data.preprocessors import MultiHotEncoder
        >>>
        >>> df = pd.DataFrame({
        ...     "name": ["Shaolin Soccer", "Moana", "The Smartest Guys in the Room"],
        ...     "genre": [
        ...         ["comedy", "action", "sports"],
        ...         ["animation", "comedy",  "action"],
        ...         ["documentary"],
        ...     ],
        ... })
        >>> ds = ray.data.from_pandas(df)  # doctest: +SKIP
        >>>
        >>> encoder = MultiHotEncoder(columns=["genre"])
        >>> encoder.fit_transform(ds).to_pandas()  # doctest: +SKIP
                                    name            genre
        0                 Shaolin Soccer  [1, 0, 1, 0, 1]
        1                          Moana  [1, 1, 1, 0, 0]
        2  The Smartest Guys in the Room  [0, 0, 0, 1, 0]

        If you specify ``max_categories``, then :class:`MultiHotEncoder`
        creates features for only the most frequent categories.

        >>> encoder = MultiHotEncoder(columns=["genre"], max_categories={"genre": 3})
        >>> encoder.fit_transform(ds).to_pandas()  # doctest: +SKIP
                                    name      genre
        0                 Shaolin Soccer  [1, 1, 1]
        1                          Moana  [1, 1, 0]
        2  The Smartest Guys in the Room  [0, 0, 0]
        >>> encoder.stats_  # doctest: +SKIP
        OrderedDict([('unique_values(genre)', {'comedy': 0, 'action': 1, 'sports': 2})])

    Args:
        columns: The columns to separately encode.
        max_categories: The maximum number of features to create for each column.
            If a value isn't specified for a column, then a feature is created
            for every unique category in that column.

    .. seealso::

        :class:`OneHotEncoder`
            If you're encoding individual categories instead of lists of
            categories, use :class:`OneHotEncoder`.

        :class:`OrdinalEncoder`
            If your categories are ordered, you may want to use
            :class:`OrdinalEncoder`.
    """

    def __init__(self, columns: List[str], *, max_categories: Optional[Dict[str, int]]=None):
        self.columns = columns
        self.max_categories = max_categories

    def _fit(self, dataset: Dataset) -> Preprocessor:
        self.stats_ = _get_unique_value_indices(dataset, self.columns, max_categories=self.max_categories, encode_lists=True)
        return self

    def _transform_pandas(self, df: pd.DataFrame):
        _validate_df(df, *self.columns)

        def encode_list(element: list, *, name: str):
            if isinstance(element, np.ndarray):
                element = element.tolist()
            elif not isinstance(element, list):
                element = [element]
            stats = self.stats_[f'unique_values({name})']
            counter = Counter(element)
            return [counter.get(x, 0) for x in stats]
        for column in self.columns:
            df[column] = df[column].map(partial(encode_list, name=column))
        return df

    def __repr__(self):
        return f'{self.__class__.__name__}(columns={self.columns!r}, max_categories={self.max_categories!r})'