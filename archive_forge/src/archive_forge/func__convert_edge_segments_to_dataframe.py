from __future__ import annotations
from math import ceil
from dask import compute, delayed
from pandas import DataFrame
import numpy as np
import pandas as pd
import param
from .utils import ngjit
def _convert_edge_segments_to_dataframe(edge_segments, segment_class, params):
    """
    Convert list of edge segments into a dataframe.

    For all edge segments, we create a dataframe to represent a path
    as successive points separated by a point with NaN as the x or y
    value.
    """

    def edge_iterator():
        for edge in edge_segments:
            yield edge
            yield segment_class.create_delimiter()
    df = DataFrame(np.concatenate(list(edge_iterator())))
    df.columns = segment_class.get_columns(params)
    return df