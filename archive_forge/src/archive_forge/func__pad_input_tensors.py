import pickle
import warnings
from functools import update_wrapper, wraps
from typing import Any, Mapping
import torch
from ..state import PartialState
from .constants import TORCH_DISTRIBUTED_OPERATION_TYPES
from .dataclasses import DistributedType, TensorInformation
from .imports import (
def _pad_input_tensors(tensor, batch_size, num_processes, dim=0):
    remainder = batch_size // num_processes
    last_inputs = batch_size - remainder * num_processes
    if batch_size // num_processes == 0:
        to_pad = num_processes - batch_size
    else:
        to_pad = num_processes - batch_size // num_processes
    if last_inputs > to_pad & to_pad < 1:
        to_pad = last_inputs - to_pad
    old_size = tensor.shape
    new_size = list(old_size)
    new_size[0] = batch_size + to_pad
    new_tensor = tensor.new_zeros(tuple(new_size))
    indices = tuple((slice(0, old_size[dim]) if i == dim else slice(None) for i in range(len(new_size))))
    new_tensor[indices] = tensor
    return new_tensor