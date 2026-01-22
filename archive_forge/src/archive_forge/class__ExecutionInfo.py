import functools
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple
import torch
import torch.nn as nn
class _ExecutionInfo:
    """
    This represents the execution order information from the forward pass.

    Attributes:
        curr_module (nn.Module): Current module being traced.
        module_forward_order (List[nn.Module]): The modules in (pre-)forward
            order, i.e. the order in which their ``forward()`` methods are
            called. Each call to a module's ``forward()`` corresponds to one
            element in the list.
        module_to_param_usage_infos (Dict[nn.Module, List[_ParamUsageInfo]]):
            Maps a module to a list of module execution infos. See
            :class:`_ParamUsageInfo` for details.
        param_forward_order (List[nn.Parameter]): The parameters in forward
            execution order, where only a parameter's first participation is
            included.
        visited_params (Set[nn.Parameter]): The parameters visited so far
            during the trace. This is only used during tracing for fast
            membership check. Invariant: The parameters in
            ``param_forward_order`` are exactly those in ``visited_params``.
    """

    def __init__(self, root_module: nn.Module) -> None:
        self.curr_module: nn.Module = root_module
        self.module_forward_order: List[nn.Module] = [root_module]
        self.module_to_param_usage_infos: Dict[nn.Module, List[_ParamUsageInfo]] = {root_module: []}
        self.param_forward_order: List[nn.Parameter] = []
        self.visited_params: Set[nn.Parameter] = set()