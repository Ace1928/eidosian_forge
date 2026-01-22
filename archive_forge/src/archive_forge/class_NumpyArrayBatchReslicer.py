import math
from typing import Any, Generic, Iterable, List, Optional, Tuple, TypeVar
import numpy as np
import pandas as pd
import pyarrow as pa
from triad.utils.convert import to_size
from .iter import slice_iterable
class NumpyArrayBatchReslicer(BatchReslicer[np.ndarray]):

    def get_rows_and_size(self, batch: np.ndarray) -> Tuple[int, int]:
        return (batch.shape[0], batch.nbytes)

    def take(self, batch: np.ndarray, start: int, length: int) -> np.ndarray:
        if start == 0 and length == batch.shape[0]:
            return batch
        return batch[start:start + length]

    def concat(self, batches: List[np.ndarray]) -> np.ndarray:
        if len(batches) == 1:
            return batches[0]
        return np.concatenate(batches, axis=0)