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
def _is_series_composed_of_lists(series: pd.Series) -> bool:
    first_not_none_element = next((element for element in series if element is not None), None)
    return pandas.api.types.is_object_dtype(series.dtype) and isinstance(first_not_none_element, (list, np.ndarray))