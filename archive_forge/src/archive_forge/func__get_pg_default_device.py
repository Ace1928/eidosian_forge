import itertools
import collections.abc
import contextlib
import hashlib
import io
import logging
import os
import pickle
import sys
import time
import warnings
from collections import namedtuple
from datetime import timedelta
from typing import Any, Callable, Dict, Optional, Tuple, Union, List
import torch
from torch._C._distributed_c10d import (
from .constants import default_pg_timeout, default_pg_nccl_timeout
from .c10d_logger import _exception_logger, _time_logger
from .rendezvous import register_rendezvous_handler, rendezvous  # noqa: F401
def _get_pg_default_device(group: Optional[ProcessGroup]=None) -> torch.device:
    """
    Return the device to use with ``group`` for control flow usage (object collectives, barrier).

    There are selection rules:
        1. If user specifies exactly one backend in ``init_process_group`` call:
            use that backend
        2. Else if user specifies multiple "device:backend" pairs in init_process_group:
            If "cpu" is among those pairs, use "cpu" (because the object is in cpu memory);
            Otherwise, use the first backend (sort of a random pick).

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.

    Returns:
        torch.device: The device to use with ``group``.

    """
    group = group or _get_default_group()
    if group in _world.pg_default_device:
        return _world.pg_default_device[group]
    if not isinstance(group, ProcessGroup):
        warnings.warn(f'You are using a Backend {type(group)} as a ProcessGroup. This usage is deprecated since PyTorch 2.0. Please use a public API of PyTorch Distributed instead.')
        _world.pg_default_device[group] = torch.device('cpu')
        return _world.pg_default_device[group]
    '\n    ``group._device_types`` is a property pybind that returns the devices\n    ("cpu", "cuda", etc) supported by ``group``. Can be multiple if the\n    ``group`` supports multiple devices.\n    '
    devices = group._device_types
    if len(devices) == 1:
        _world.pg_default_device[group] = devices[0]
    elif len(devices) == 0:
        _world.pg_default_device[group] = torch.device('cpu')
    elif torch.device('cpu') in devices:
        _world.pg_default_device[group] = torch.device('cpu')
    else:
        _world.pg_default_device[group] = devices[0]
    logger.info(f'Using device {_world.pg_default_device[group]} for object collectives.')
    return _world.pg_default_device[group]