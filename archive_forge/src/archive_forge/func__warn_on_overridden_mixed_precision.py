import collections
import functools
import inspect
import warnings
from functools import partial
from typing import Any, Callable, Dict, List, Set, Tuple, Type, Union
import torch.nn as nn
from torch.distributed.fsdp._common_utils import (
from torch.distributed.fsdp.wrap import (
def _warn_on_overridden_mixed_precision(overridden_module_classes: Set[Type[nn.Module]]):
    if len(overridden_module_classes) == 0:
        return
    warnings.warn(f'Both mixed precision and an auto_wrap_policy were specified to FSDP, where the wrapped module has submodules of type:\n{overridden_module_classes}\nThese modules will be wrapped as separate FSDP instacnes with mixed precision disabled.')