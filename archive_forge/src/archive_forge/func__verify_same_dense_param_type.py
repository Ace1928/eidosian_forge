import collections
import copy
import enum
import inspect
import io
import logging
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union
import torch
import torch.distributed as dist
from torch.distributed.algorithms.join import Join, Joinable, JoinHook
from torch.distributed.optim.utils import functional_optim_map
from torch.optim import Optimizer
def _verify_same_dense_param_type(self) -> None:
    """
        Verify that all parameters are of the same dense type.

        The method assumes that ``self._all_params`` has been initialized
        and is non-empty.

        Raises:
            ValueError: ``params`` contains sparse parameters or parameters
            of varying dense types.

        NOTE: This method can be removed once support for sparse parameters
        and varying parameter types is added.
        """
    typename = torch.typename(self._all_params[0])
    if self._all_params[0].is_sparse:
        raise ValueError(f'ZeroRedundancyOptimizer only supports using the same dense type for all parameters but got {typename}')
    for param in self._all_params[1:]:
        other_typename = torch.typename(param)
        if other_typename != typename:
            raise ValueError(f'ZeroRedundancyOptimizer only supports using the same dense type for all parameters but got both {typename} and {other_typename}')