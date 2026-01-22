from __future__ import annotations
from math import ceil
from dask import compute, delayed
from pandas import DataFrame
import numpy as np
import pandas as pd
import param
from .utils import ngjit
def advect_and_resample(vert, horiz, segments, iterations, accuracy, min_segment_length, max_segment_length, segment_class):
    for it in range(iterations):
        advect_segments(segments, vert, horiz, accuracy, segment_class.idx, segment_class.idy)
        if it % 2 == 0:
            segments = resample_edge(segments, min_segment_length, max_segment_length, segment_class.ndims)
    return segments