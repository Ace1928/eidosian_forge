from typing import Dict, Iterable, List, Optional, Type, Union
import numpy as np
import pandas as pd
from ray.data import Dataset
from ray.data.aggregate import Max, Min
from ray.data.preprocessor import Preprocessor
from ray.util.annotations import PublicAPI
def _validate_bins_columns(self):
    if isinstance(self.bins, dict) and (not all((col in self.bins for col in self.columns))):
        raise ValueError('If `bins` is a dictionary, all elements of `columns` must be present in it.')