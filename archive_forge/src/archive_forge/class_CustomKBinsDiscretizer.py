from typing import Dict, Iterable, List, Optional, Type, Union
import numpy as np
import pandas as pd
from ray.data import Dataset
from ray.data.aggregate import Max, Min
from ray.data.preprocessor import Preprocessor
from ray.util.annotations import PublicAPI
@PublicAPI(stability='alpha')
class CustomKBinsDiscretizer(_AbstractKBinsDiscretizer):
    """Bin values into discrete intervals using custom bin edges.

    Columns must contain numerical values.

    Examples:
        Use :class:`CustomKBinsDiscretizer` to bin continuous features.

        >>> import pandas as pd
        >>> import ray
        >>> from ray.data.preprocessors import CustomKBinsDiscretizer
        >>> df = pd.DataFrame({
        ...     "value_1": [0.2, 1.4, 2.5, 6.2, 9.7, 2.1],
        ...     "value_2": [10, 15, 13, 12, 23, 25],
        ... })
        >>> ds = ray.data.from_pandas(df)
        >>> discretizer = CustomKBinsDiscretizer(
        ...     columns=["value_1", "value_2"],
        ...     bins=[0, 1, 4, 10, 25]
        ... )
        >>> discretizer.transform(ds).to_pandas()
           value_1  value_2
        0        0        2
        1        1        3
        2        1        3
        3        2        3
        4        2        3
        5        1        3

        You can also specify different bin edges per column.

        >>> discretizer = CustomKBinsDiscretizer(
        ...     columns=["value_1", "value_2"],
        ...     bins={"value_1": [0, 1, 4], "value_2": [0, 18, 35, 70]},
        ... )
        >>> discretizer.transform(ds).to_pandas()
           value_1  value_2
        0      0.0        0
        1      1.0        0
        2      1.0        0
        3      NaN        0
        4      NaN        1
        5      1.0        1


    Args:
        columns: The columns to discretize.
        bins: Defines custom bin edges. Can be an iterable of numbers,
            a ``pd.IntervalIndex``, or a dict mapping columns to either of them.
            Note that ``pd.IntervalIndex`` for bins must be non-overlapping.
        right: Indicates whether bins include the rightmost edge.
        include_lowest: Indicates whether the first interval should be left-inclusive.
        duplicates: Can be either 'raise' or 'drop'. If bin edges are not unique,
            raise ``ValueError`` or drop non-uniques.
        dtypes: An optional dictionary that maps columns to ``pd.CategoricalDtype``
            objects or ``np.integer`` types. If you don't include a column in ``dtypes``
            or specify it as an integer dtype, the outputted column will consist of
            ordered integers corresponding to bins. If you use a
            ``pd.CategoricalDtype``, the outputted column will be a
            ``pd.CategoricalDtype`` with the categories being mapped to bins.
            You can use ``pd.CategoricalDtype(categories, ordered=True)`` to
            preserve information about bin order.

    .. seealso::

        :class:`UniformKBinsDiscretizer`
            If you want to bin data into uniform width bins.
    """

    def __init__(self, columns: List[str], bins: Union[Iterable[float], pd.IntervalIndex, Dict[str, Union[Iterable[float], pd.IntervalIndex]]], *, right: bool=True, include_lowest: bool=False, duplicates: str='raise', dtypes: Optional[Dict[str, Union[pd.CategoricalDtype, Type[np.integer]]]]=None):
        self.columns = columns
        self.bins = bins
        self.right = right
        self.include_lowest = include_lowest
        self.duplicates = duplicates
        self.dtypes = dtypes
        self._validate_bins_columns()
    _is_fittable = False