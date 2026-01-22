import functools
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple
import torch
import torch.nn as nn
class _ParamUsageInfo(NamedTuple):
    """
    This is used for ``_ExecutionInfo.module_to_param_usage_infos`` to record
    execution information. The ``dict`` maps modules to a list of these
    ``_ParamUsageInfo`` instances, where each instance represents a group of
    parameters used together.

    Specifically, for each module key in the ``dict``, each instance of this
    class represents either:
    (1) the module and some sublist of its ``named_parameters()`` used
    together in execution (see ``_patched_create_proxy()``), or
    (2) a submodule and all of ``submodule.named_parameters()`` (see
    ``_patched_call_module()``).

    Type (1) corresponds to directly using parameters in ops without calling
    ``forward()``, and type (2) corresponds to calling ``forward()``. The
    mapped-to lists in the ``dict`` follow the execution order.
    """
    module: nn.Module
    named_params: List[Tuple[str, nn.Parameter]]