from __future__ import annotations
import numpy as np
import pandas as pd
from pandas.core.resample import Resampler as pd_Resampler
from dask.base import tokenize
from dask.dataframe import methods
from dask.dataframe._compat import PANDAS_GE_140
from dask.dataframe.core import DataFrame, Series
from dask.highlevelgraph import HighLevelGraph
from dask.utils import derived_from
@derived_from(pd_Resampler)
def agg(self, agg_funcs, *args, **kwargs):
    return self._agg('agg', how_args=(agg_funcs,) + args, how_kwargs=kwargs)