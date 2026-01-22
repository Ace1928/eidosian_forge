from typing import Callable, Iterator, Tuple, TypeVar
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.dataframe import dataframe_wrapper as df_wrapper
from torch.utils.data.datapipes.utils.common import (
def _apply_filter_fn(self, data) -> bool:
    if self.input_col is None:
        return self.filter_fn(data)
    elif isinstance(self.input_col, (list, tuple)):
        args = tuple((data[col] for col in self.input_col))
        return self.filter_fn(*args)
    else:
        return self.filter_fn(data[self.input_col])