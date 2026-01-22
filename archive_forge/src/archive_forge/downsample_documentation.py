import math
from functools import partial
import numpy as np
import param
from ..core import NdOverlay, Overlay
from ..element.chart import Area
from .resample import ResampleOperation1D

    Implements downsampling of a regularly sampled 1D dataset.

    Supports multiple algorithms:

        - `lttb`: Largest Triangle Three Buckets downsample algorithm
        - `nth`: Selects every n-th point.
    