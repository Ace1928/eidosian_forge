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
def _get_optimizer_constructor(self, optimizer_class: Any) -> Any:
    """
        Return the optimizer constructor using validation and transformation depending on ``overlap_with_ddp``.

        Returns:
            - ``optimizer_class`` if ``overlap_with_ddp=False`` and
                ``optimizer_class`` is not a functional optimizer.
            - ``optimizer_class`` if ``overlap_with_ddp=True`` and
                ``optimizer_class`` is already a functional optimizer.
            - The functional equivalent of ``optimizer_class`` if
                ``overlap_with_ddp=True`` and ``optimizer_class`` is not
                already a functional optimizer (assuming the equivalent
                exists).

        Raises:
            ValueError:

                - if ``overlap_with_ddp=True`` but ``optimizer_class`` is
                    neither a functional optimizer nor translatable to a
                    functional optimizer.
                - if ``overlap_with_ddp=False`` and ``optimizer_class`` is a
                    functional optimizer.
        """
    functional_optims = functional_optim_map.values()
    if not self._overlap_with_ddp:
        if optimizer_class in functional_optims:
            raise ValueError(f'Passing in a functional optimizer {optimizer_class} when `overlap_with_ddp=False`')
        else:
            return optimizer_class
    elif optimizer_class in functional_optims:
        return optimizer_class
    elif optimizer_class in functional_optim_map:
        optim_constructor = functional_optim_map[optimizer_class]
        logger.info('Using the functional optimizer %s instead of %s since `overlap_with_ddp=True`', optim_constructor, optimizer_class)
        return optim_constructor
    else:
        raise ValueError(f'Using `ddp_with_overlap=True` requires using a functional optimizer, but there is no supported functional optimizer equivalent for {optimizer_class}')