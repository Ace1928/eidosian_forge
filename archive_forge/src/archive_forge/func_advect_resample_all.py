from __future__ import annotations
from math import ceil
from dask import compute, delayed
from pandas import DataFrame
import numpy as np
import pandas as pd
import param
from .utils import ngjit
@delayed
def advect_resample_all(gradients, edge_segments, iterations, accuracy, min_segment_length, max_segment_length, segment_class):
    vert, horiz = gradients
    return [advect_and_resample(vert, horiz, edges, iterations, accuracy, min_segment_length, max_segment_length, segment_class) for edges in edge_segments]