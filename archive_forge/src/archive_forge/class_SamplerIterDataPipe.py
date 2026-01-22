import random
import torch
from torch.utils.data import Sampler, SequentialSampler
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe
from typing import Dict, Iterator, List, Optional, Sized, Tuple, Type, TypeVar
class SamplerIterDataPipe(IterDataPipe[T_co]):
    """
    Generate sample elements using the provided ``Sampler`` (defaults to :class:`SequentialSampler`).

    Args:
        datapipe: IterDataPipe to sample from
        sampler: Sampler class to generate sample elements from input DataPipe.
            Default is :class:`SequentialSampler` for IterDataPipe
    """
    datapipe: IterDataPipe
    sampler: Sampler

    def __init__(self, datapipe: IterDataPipe, sampler: Type[Sampler]=SequentialSampler, sampler_args: Optional[Tuple]=None, sampler_kwargs: Optional[Dict]=None) -> None:
        assert isinstance(datapipe, Sized), 'Sampler class requires input datapipe implemented `__len__`'
        super().__init__()
        self.datapipe = datapipe
        self.sampler_args = () if sampler_args is None else sampler_args
        self.sampler_kwargs = {} if sampler_kwargs is None else sampler_kwargs
        self.sampler = sampler(*self.sampler_args, data_source=self.datapipe, **self.sampler_kwargs)

    def __iter__(self) -> Iterator[T_co]:
        return iter(self.sampler)

    def __len__(self) -> int:
        if isinstance(self.sampler, Sized):
            return len(self.sampler)
        raise TypeError(f"{type(self).__name__} instance doesn't have valid length")