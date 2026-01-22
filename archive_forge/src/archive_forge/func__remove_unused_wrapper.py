import torch.fx as fx
import copy
import torch
import math
import sys
from typing import Callable, List
from functools import wraps, partial
from dataclasses import dataclass
from .compile_utils import get_placeholders, get_outputs
from torch.utils._content_store import ContentStoreWriter
from torch.hub import tqdm
from torch.multiprocessing.reductions import StorageWeakRef
import os
def _remove_unused_wrapper(cur_graph, cur_inps, granularity):
    return remove_unused_inputs_checked(ReproState(cur_graph, cur_inps))