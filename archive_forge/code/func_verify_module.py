from collections import OrderedDict
from dataclasses import dataclass, field
import itertools
import threading
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union
import warnings
import torch
from torch import Tensor, nn
from fairscale.nn.model_parallel import get_pipeline_parallel_group
from . import microbatch
from .async_pipeline import AsyncPipeline
from .async_schedule import Invocation, Location, ModuleWrapper
from .batchnorm import DeferredBatchNorm
from .skip.layout import SkipLayout
from .skip.skippable import Skippable
from .types import LazyModule
def verify_module(module: Union[nn.Sequential, List[LazyModule]]) -> None:
    if len(set(map(id, module))) != len(module):
        raise ValueError('module with duplicate children is not supported')