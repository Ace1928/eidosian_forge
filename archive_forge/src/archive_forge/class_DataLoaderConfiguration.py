import argparse
import copy
import enum
import functools
import os
import typing
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, get_args
import torch
from .constants import FSDP_AUTO_WRAP_POLICY, FSDP_BACKWARD_PREFETCH, FSDP_SHARDING_STRATEGY, FSDP_STATE_DICT_TYPE
from .environment import str_to_bool
from .imports import is_cuda_available, is_npu_available, is_xpu_available
from .versions import compare_versions
@dataclass
class DataLoaderConfiguration:
    """
    Configuration for dataloader-related items when calling `accelerator.prepare`.
    """
    split_batches: bool = field(default=False, metadata={'help': 'Whether or not the accelerator should split the batches yielded by the dataloaders across the devices. If `True` the actual batch size used will be the same on any kind of distributed processes, but it must be a round multiple of the `num_processes` you are using. If `False`, actual batch size used will be the one set in your script multiplied by the number of processes.'})
    dispatch_batches: bool = field(default=None, metadata={'help': 'If set to `True`, the dataloader prepared by the Accelerator is only iterated through on the main process and then the batches are split and broadcast to each process. Will default to `True` for `DataLoader` whose underlying dataset is an `IterableDataslet`, `False` otherwise.'})
    even_batches: bool = field(default=True, metadata={'help': 'If set to `True`, in cases where the total batch size across all processes does not exactly divide the dataset, samples at the start of the dataset will be duplicated so the batch can be divided equally among all workers.'})
    use_seedable_sampler: bool = field(default=False, metadata={'help': 'Whether or not use a fully seedable random sampler ([`data_loader.SeedableRandomSampler`]).Ensures training results are fully reproducable using a different sampling technique. While seed-to-seed results may differ, on average the differences are neglible when usingmultiple different seeds to compare. Should also be ran with [`~utils.set_seed`] for the best results.'})