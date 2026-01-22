from __future__ import annotations
import math
from typing import Hashable, Sequence, Type
from pandas import DataFrame
from torch.utils.data import Sampler, SequentialSampler
from modin.pandas import DataFrame as ModinDataFrame
def _end_of_batch(self, counter: int):
    return counter % self._batch_size == self._batch_size - 1 or counter == len(self._sampler) - 1