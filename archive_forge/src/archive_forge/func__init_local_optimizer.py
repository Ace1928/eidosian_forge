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
def _init_local_optimizer(self) -> None:
    """
        Initialize this rank's local optimizer, responsible for its subset of the parameters.

        The local optimizer is saved in ``self.optim``.
        """
    assert self._optim_constructor is not None, 'The local optimizer class has not been set'
    param_groups = self._partition_parameters()[self.rank]
    if self._overlap_with_ddp:
        assert len(param_groups) == 1, 'Initializing the local functional optimizer with more than one parameter group'
        params = param_groups[0]['params']
        if '_allow_empty_param_list' in inspect.signature(self._optim_constructor).parameters:
            self.optim: Any = self._optim_constructor(params, **self._optim_defaults, _allow_empty_param_list=True)
        else:
            logger.warning('%s does not support the argument `_allow_empty_param_list`; ZeroRedundancyOptimizer may error due to an empty parameter list', self._optim_constructor)
            self.optim: Any = self._optim_constructor(params, **self._optim_defaults)
        if dist.get_debug_level() != dist.DebugLevel.OFF:
            local_numel = sum((p.numel() for p in params))
            num_assigned_buckets = len(self._bucket_assignments_per_rank[self.global_rank])
            logger.info('rank %s with %s parameters across %s buckets', self.global_rank, local_numel, num_assigned_buckets)
            if self.global_rank == 0:
                logger.info('%s DDP buckets and %s bucket assignments', len(self._overlap_info.params_per_bucket), self._overlap_info.num_bucket_assignments)
    else:
        self.optim: Optimizer = self._optim_constructor(param_groups, **self._optim_defaults)
    if self._overlap_with_ddp and (not hasattr(self.optim, 'param_groups')):
        assert hasattr(self.optim, 'param_group'), 'The functional optimizer should set at least one of the attributes `param_group` or `param_groups`'
        self.optim.param_groups = [self.optim.param_group]
    self._sync_param_groups(self.optim.param_groups, self.param_groups)