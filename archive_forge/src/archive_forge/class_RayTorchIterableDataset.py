import logging
import os
import shutil
from tempfile import TemporaryDirectory
from typing import Iterator, Optional, Type
from torch.utils.data import DataLoader, Dataset, IterableDataset
import ray
from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
from ray.data.iterator import _IterableFromIterator
from ray.train import Checkpoint
from ray.util import PublicAPI
class RayTorchIterableDataset(IterableDataset):
    """Wrapper class for ray data iterables."""

    def __init__(self, data_iterable) -> None:
        super().__init__()
        self.data_iterable = data_iterable

    def __iter__(self) -> Iterator:
        return iter(self.data_iterable)