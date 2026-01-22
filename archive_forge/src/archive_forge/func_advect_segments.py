from __future__ import annotations
from math import ceil
from dask import compute, delayed
from pandas import DataFrame
import numpy as np
import pandas as pd
import param
from .utils import ngjit
@ngjit
def advect_segments(segments, vert, horiz, accuracy, idx, idy):
    for i in range(1, len(segments) - 1):
        x = int(segments[i][idx] * accuracy)
        y = int(segments[i][idy] * accuracy)
        segments[i][idx] = segments[i][idx] + horiz[x, y] / accuracy
        segments[i][idy] = segments[i][idy] + vert[x, y] / accuracy
        segments[i][idx] = max(0, min(segments[i][idx], 1))
        segments[i][idy] = max(0, min(segments[i][idy], 1))