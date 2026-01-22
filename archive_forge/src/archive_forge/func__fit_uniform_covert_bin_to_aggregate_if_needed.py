from typing import Dict, Iterable, List, Optional, Type, Union
import numpy as np
import pandas as pd
from ray.data import Dataset
from ray.data.aggregate import Max, Min
from ray.data.preprocessor import Preprocessor
from ray.util.annotations import PublicAPI
def _fit_uniform_covert_bin_to_aggregate_if_needed(self, column: str):
    bins = self.bins[column] if isinstance(self.bins, dict) else self.bins
    if isinstance(bins, int):
        return (Min(column), Max(column))
    else:
        raise TypeError(f'`bins` must be an integer or a dict of integers, got {bins}')