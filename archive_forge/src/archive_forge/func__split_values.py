from __future__ import annotations
import logging
import math
import os
import threading
import warnings
from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, Optional
import torch
from .utils import (
from .utils.dataclasses import SageMakerDistributedType
def _split_values(inputs, start_index, end_index):
    if isinstance(inputs, (list, tuple, torch.Tensor)):
        if start_index >= len(inputs):
            result = inputs[-1:]
        else:
            result = inputs[start_index:end_index]
        if apply_padding:
            if isinstance(result, torch.Tensor):
                from accelerate.utils import pad_across_processes, send_to_device
                tensorized_result = send_to_device(result, self.device)
                result = pad_across_processes(tensorized_result, pad_index=inputs[-1])
            else:
                result += [result[-1]] * (num_samples_per_process - len(result))
        return result
    elif isinstance(inputs, dict):
        for key in inputs.keys():
            inputs[key] = _split_values(inputs[key], start_index, end_index)
        return inputs
    else:
        return inputs