import contextlib
import logging
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator, List, Optional, Sized, Union
import torch
import torch.nn.functional as F
from lightning_utilities.core.imports import package_available
from torch import Tensor
from torch.utils.data import Dataset, DistributedSampler, Sampler
from typing_extensions import override
from lightning_fabric.utilities.cloud_io import _is_local_file_protocol
from lightning_fabric.utilities.data import _num_cpus_available
from lightning_fabric.utilities.rank_zero import rank_zero_info
from lightning_fabric.utilities.types import _PATH, ReduceOp
class _DatasetSamplerWrapper(Dataset):
    """Dataset to create indexes from `Sampler` or `Iterable`"""

    def __init__(self, sampler: Union[Sampler, Iterable]) -> None:
        if not isinstance(sampler, Sized):
            raise TypeError('You seem to have configured a sampler in your DataLoader which does not provide `__len__` method. The sampler was about to be replaced by `DistributedSamplerWrapper` since `use_distributed_sampler` is True and you are using distributed training. Either provide `__len__` method in your sampler, remove it from DataLoader or set `use_distributed_sampler=False` if you want to handle distributed sampling yourself.')
        if len(sampler) == float('inf'):
            raise TypeError('You seem to have configured a sampler in your DataLoader which does not provide finite `__len__` method. The sampler was about to be replaced by `DistributedSamplerWrapper` since `use_distributed_sampler` is True and you are using distributed training. Either provide `__len__` method in your sampler which returns a finite number, remove it from DataLoader or set `use_distributed_sampler=False` if you want to handle distributed sampling yourself.')
        self._sampler = sampler
        self._sampler_list: Optional[List[Any]] = None

    @override
    def __getitem__(self, index: int) -> Any:
        if self._sampler_list is None:
            self._sampler_list = list(self._sampler)
        return self._sampler_list[index]

    def __len__(self) -> int:
        return len(self._sampler)

    def reset(self) -> None:
        """Reset the sampler list in order to get new sampling."""
        self._sampler_list = list(self._sampler)