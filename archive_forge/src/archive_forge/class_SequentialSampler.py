import torch
from torch import Tensor
from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union
class SequentialSampler(Sampler[int]):
    """Samples elements sequentially, always in the same order.

    Args:
        data_source (Dataset): dataset to sample from
    """
    data_source: Sized

    def __init__(self, data_source: Sized) -> None:
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)