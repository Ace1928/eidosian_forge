from __future__ import annotations
from math import ceil
from dask import compute, delayed
from pandas import DataFrame
import numpy as np
import pandas as pd
import param
from .utils import ngjit
class BaseSegment:

    @classmethod
    def create_delimiter(cls):
        return np.full((1, cls.ndims), np.nan)