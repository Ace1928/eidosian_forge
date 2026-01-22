import functools
import inspect
import os
from collections import OrderedDict
from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, Dict, Generator, Iterable, Optional, Sized, Tuple, Type, Union
from lightning_utilities.core.inheritance import get_all_subclasses
from torch.utils.data import BatchSampler, DataLoader, IterableDataset, Sampler
from typing_extensions import TypeGuard
from lightning_fabric.utilities.enums import LightningEnum
from lightning_fabric.utilities.exceptions import MisconfigurationException
from lightning_fabric.utilities.rank_zero import rank_zero_warn
from lightning_fabric.utilities.seed import pl_worker_init_function
def _set_sampler_epoch(dataloader: object, epoch: int) -> None:
    """Calls the ``set_epoch`` method on either the sampler of the given dataloader.

    Every PyTorch dataloader has either a sampler or a batch sampler. If the sampler is wrapped by a
    :class:`~torch.utils.data.distributed.DistributedSampler`, ``set_epoch`` must be called at the beginning
    of every epoch to ensure shuffling applies a new ordering. This has no effect if shuffling is off.

    """
    objects: Dict[int, Any] = {}
    if (sampler := getattr(dataloader, 'sampler', None)) is not None:
        objects[id(sampler)] = sampler
    if (batch_sampler := getattr(dataloader, 'batch_sampler', None)) is not None and (sampler := getattr(batch_sampler, 'sampler', None)) is not None:
        objects[id(sampler)] = sampler
    for obj in objects.values():
        set_epoch = getattr(obj, 'set_epoch', None)
        if callable(set_epoch):
            set_epoch(epoch)