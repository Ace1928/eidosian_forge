from typing import Dict, Iterable, List, Optional, Type, Union
import numpy as np
import pandas as pd
from ray.data import Dataset
from ray.data.aggregate import Max, Min
from ray.data.preprocessor import Preprocessor
from ray.util.annotations import PublicAPI
class _AbstractKBinsDiscretizer(Preprocessor):
    """Abstract base class for all KBinsDiscretizers.

    Essentially a thin wraper around ``pd.cut``.

    Expects either ``self.stats_`` or ``self.bins`` to be set and
    contain {column:list_of_bin_intervals}.
    """

    def _transform_pandas(self, df: pd.DataFrame):

        def bin_values(s: pd.Series) -> pd.Series:
            if s.name not in self.columns:
                return s
            labels = self.dtypes.get(s.name) if self.dtypes else False
            ordered = True
            if labels:
                if isinstance(labels, pd.CategoricalDtype):
                    ordered = labels.ordered
                    labels = list(labels.categories)
                else:
                    labels = False
            bins = self.stats_ if self._is_fittable else self.bins
            return pd.cut(s, bins[s.name] if isinstance(bins, dict) else bins, right=self.right, labels=labels, ordered=ordered, retbins=False, include_lowest=self.include_lowest, duplicates=self.duplicates)
        return df.apply(bin_values, axis=0)

    def _validate_bins_columns(self):
        if isinstance(self.bins, dict) and (not all((col in self.bins for col in self.columns))):
            raise ValueError('If `bins` is a dictionary, all elements of `columns` must be present in it.')

    def __repr__(self):
        attr_str = ', '.join([f'{attr_name}={attr_value!r}' for attr_name, attr_value in vars(self).items() if not attr_name.startswith('_')])
        return f'{self.__class__.__name__}({attr_str})'