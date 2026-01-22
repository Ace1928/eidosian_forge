from __future__ import annotations
import math
import pickle
import warnings
from functools import partial, wraps
import numpy as np
import pandas as pd
from pandas.api.types import is_dtype_equal
from tlz import merge_sorted, unique
from dask.base import is_dask_collection, tokenize
from dask.dataframe import methods
from dask.dataframe.core import (
from dask.dataframe.dispatch import group_split_dispatch, hash_object_dispatch
from dask.dataframe.io import from_pandas
from dask.dataframe.shuffle import (
from dask.dataframe.utils import (
from dask.highlevelgraph import HighLevelGraph
from dask.layers import BroadcastJoinLayer
from dask.utils import M, apply, get_default_shuffle_method
def _recursive_pairwise_outer_join(dataframes_to_merge, on, lsuffix, rsuffix, npartitions, shuffle_method):
    """
    Schedule the merging of a list of dataframes in a pairwise method. This is a recursive function that results
    in a much more efficient scheduling of merges than a simple loop
    from:
    [A] [B] [C] [D] -> [AB] [C] [D] -> [ABC] [D] -> [ABCD]
    to:
    [A] [B] [C] [D] -> [AB] [CD] -> [ABCD]
    Note that either way, n-1 merges are still required, but using a pairwise reduction it can be completed in parallel.
    :param dataframes_to_merge: A list of Dask dataframes to be merged together on their index
    :return: A single Dask Dataframe, comprised of the pairwise-merges of all provided dataframes
    """
    number_of_dataframes_to_merge = len(dataframes_to_merge)
    merge_options = {'on': on, 'lsuffix': lsuffix, 'rsuffix': rsuffix, 'npartitions': npartitions, 'shuffle_method': shuffle_method}
    if number_of_dataframes_to_merge == 1:
        return dataframes_to_merge[0]
    if number_of_dataframes_to_merge == 2:
        merged_ddf = dataframes_to_merge[0].join(dataframes_to_merge[1], how='outer', **merge_options)
        return merged_ddf
    else:
        middle_index = number_of_dataframes_to_merge // 2
        merged_ddf = _recursive_pairwise_outer_join([_recursive_pairwise_outer_join(dataframes_to_merge[:middle_index], **merge_options), _recursive_pairwise_outer_join(dataframes_to_merge[middle_index:], **merge_options)], **merge_options)
        return merged_ddf