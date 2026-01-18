import collections
import logging
import os
import random
import types
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import torch
from packaging.version import Version
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import (
from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
from ray.train._internal import session
from ray.train._internal.accelerator import Accelerator
from ray.train._internal.session import get_accelerator, set_accelerator
from ray.util.annotations import Deprecated, PublicAPI
def with_sampler(loader):
    shuffle = not isinstance(loader.sampler, SequentialSampler)

    def seeded_worker_init_fn(worker_init_fn: Optional[Callable[[int], None]]):

        def wrapper(worker_id: int):
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            if worker_init_fn:
                worker_init_fn(worker_id)
        return wrapper
    worker_init_fn: Optional[Callable[[int], None]] = loader.worker_init_fn
    generator: Optional[torch.Generator] = loader.generator
    if self._seed is not None:
        worker_init_fn = seeded_worker_init_fn(worker_init_fn)
        generator = torch.Generator()
        generator.manual_seed(self._seed)
    using_default_sampler = isinstance(loader.sampler, (SequentialSampler, RandomSampler))
    if not using_default_sampler and world_rank == 0:
        logger.warn(f'The {loader.sampler.__class__.__name__} will be overwritten with a DistributedSampler. You can disable this by setting `with_sampler` to False in `prepare_data_loader`.')
    data_loader_args = {'dataset': loader.dataset, 'batch_size': loader.batch_size, 'shuffle': False, 'num_workers': loader.num_workers, 'collate_fn': loader.collate_fn, 'pin_memory': loader.pin_memory, 'drop_last': loader.drop_last, 'timeout': loader.timeout, 'worker_init_fn': worker_init_fn, 'generator': generator, 'sampler': DistributedSampler(loader.dataset, shuffle=shuffle)}
    return DataLoader(**data_loader_args)