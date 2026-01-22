from typing import List
import torch
from . import config, ir, scheduler
from .dependencies import WeakDep
from .utils import tuple_sorted

        Schedules all nodes in `snodes` in an arbitrary topologically valid order.
        