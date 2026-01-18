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

    Minimizes a FX graph with given inputs, such that the resulting FX graph still returns True for module_fails.

    Does 2 main strategies:
    1. Truncates suffix: Removes some suffix from the graph and sets a new output.
    2. Delta Debugging: Tries replacing half of the graph with inputs. If fails,
        tries replacing quarter of the graph, etc.

    >>> # xdoctest: +SKIP(failing)
    >>> failing_function = fx.symbolic_trace(f)
    >>> minimize(failing_function, [torch.randn(5)], lambda fx_g, inps: fx_g(*inps))

    note: module_fails returns True if it fails.
    