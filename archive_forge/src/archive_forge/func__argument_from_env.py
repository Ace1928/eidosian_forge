import os
from collections import Counter
from typing import Any, Dict, List, Optional, Union, cast
import torch
from typing_extensions import get_args
from lightning_fabric.accelerators import ACCELERATOR_REGISTRY
from lightning_fabric.accelerators.accelerator import Accelerator
from lightning_fabric.accelerators.cuda import CUDAAccelerator
from lightning_fabric.accelerators.mps import MPSAccelerator
from lightning_fabric.accelerators.xla import XLAAccelerator
from lightning_fabric.plugins import (
from lightning_fabric.plugins.environments import (
from lightning_fabric.plugins.precision.double import DoublePrecision
from lightning_fabric.plugins.precision.fsdp import FSDPPrecision
from lightning_fabric.plugins.precision.precision import (
from lightning_fabric.strategies import (
from lightning_fabric.strategies.ddp import _DDP_FORK_ALIASES
from lightning_fabric.strategies.fsdp import _FSDP_ALIASES, FSDPStrategy
from lightning_fabric.utilities import rank_zero_info, rank_zero_warn
from lightning_fabric.utilities.device_parser import _determine_root_gpu_device
from lightning_fabric.utilities.imports import _IS_INTERACTIVE
@staticmethod
def _argument_from_env(name: str, current: Any, default: Any) -> Any:
    env_value: Optional[str] = os.environ.get('LT_' + name.upper())
    if env_value is None:
        return current
    if env_value is not None and env_value != str(current) and (str(current) != str(default)) and _is_using_cli():
        raise ValueError(f'Your code has `Fabric({name}={current!r}, ...)` but it conflicts with the value `--{name}={env_value}` set through the CLI.  Remove it either from the CLI or from the Lightning Fabric object.')
    return env_value