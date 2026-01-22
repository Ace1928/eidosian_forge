from __future__ import annotations
from collections import namedtuple
from itertools import starmap
from multiprocessing import Pipe, Process, current_process
from time import sleep
from timeit import default_timer
from dask.callbacks import Callback
from dask.utils import import_required
def _plot(self, **kwargs):
    from dask.diagnostics.profile_visualize import plot_cache
    return plot_cache(self.results, self._dsk, self.start_time, self.end_time, self._metric_name, **kwargs)