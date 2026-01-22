import torch
import functools
import threading
from torch import Tensor
from typing import Any, Callable, Optional, Tuple, Union, List
from torch.utils._pytree import (
from functools import partial
import os
import itertools
from torch._C._functorch import (
def _flatten_chunks_output(chunks_output_):
    flat_chunks_output = []
    arg_spec = None
    for output in chunks_output_:
        flat_output, arg_specs = tree_flatten(output)
        flat_chunks_output.append(flat_output)
        if arg_spec is None:
            arg_spec = arg_specs
    flat_output_chunks = list(zip(*flat_chunks_output))
    return (flat_output_chunks, arg_spec)