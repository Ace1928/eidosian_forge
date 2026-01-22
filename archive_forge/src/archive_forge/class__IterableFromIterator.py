import abc
import time
from typing import (
import numpy as np
from ray.data._internal.block_batching.iter_batches import iter_batches
from ray.data._internal.stats import DatasetStats, StatsManager
from ray.data.block import (
from ray.types import ObjectRef
from ray.util.annotations import PublicAPI
class _IterableFromIterator(Iterable[T]):

    def __init__(self, iterator_gen: Callable[[], Iterator[T]]):
        """Constructs an Iterable from an iterator generator.

        Args:
            iterator_gen: A function that returns an iterator each time it
                is called. For example, this can be a generator function.
        """
        self.iterator_gen = iterator_gen

    def __iter__(self):
        return self.iterator_gen()