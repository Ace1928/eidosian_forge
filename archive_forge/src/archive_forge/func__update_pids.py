from __future__ import annotations
from collections import namedtuple
from itertools import starmap
from multiprocessing import Pipe, Process, current_process
from time import sleep
from timeit import default_timer
from dask.callbacks import Callback
from dask.utils import import_required
def _update_pids(self, pid):
    return [self.parent] + [p for p in self.parent.children() if p.pid != pid and p.status() != 'zombie']