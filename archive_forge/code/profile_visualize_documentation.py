from __future__ import annotations
import random
import warnings
from bisect import bisect_left
from itertools import cycle
from operator import add, itemgetter
from tlz import accumulate, groupby, pluck, unique
from dask.core import istask
from dask.utils import apply, funcname, import_required
Visualize the results of profiling in a bokeh plot.

    Parameters
    ----------
    results : sequence
        Output of CacheProfiler.results
    dsk : dict
        The dask graph being profiled.
    start_time : float
        Start time of the profile in seconds
    end_time : float
        End time of the profile in seconds
    metric_name : string
        Metric used to measure cache size
    palette : string, optional
        Name of the bokeh palette to use, must be a member of
        bokeh.palettes.all_palettes.
    label_size: int (optional)
        Maximum size of output labels in plot, defaults to 60
    **kwargs
        Other keyword arguments, passed to bokeh.figure. These will override
        all defaults set by visualize.

    Returns
    -------
    The completed bokeh plot object.
    