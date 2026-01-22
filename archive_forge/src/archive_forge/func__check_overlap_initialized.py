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
def _check_overlap_initialized(self):
    """
        Check the delayed initialization depending on the value of ``overlap_with_ddp``.

        The delayed initialization has occurred (see
        :meth:`_init_zero_for_overlap`) if ``overlap_with_ddp=True``, and
        raises a ``RuntimeError`` if not. This should preface methods that
        should not be run before that delayed initialization.

        Raises:
            RuntimeError: if ``overlap_with_ddp=True`` and
                :meth:`_init_zero_for_overlap` has not been called.
        """
    if self._overlap_with_ddp and self._overlap_info.status != _OverlapStatus.INITIALIZED:
        raise RuntimeError('This method should not be called until this ZeroRedundancyOptimizer instance has been fully initialized')