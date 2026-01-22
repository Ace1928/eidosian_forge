from typing import Callable, Iterator, Tuple, TypeVar
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.dataframe import dataframe_wrapper as df_wrapper
from torch.utils.data.datapipes.utils.common import (
def _returnIfTrue(self, data: T) -> Tuple[bool, T]:
    condition = self._apply_filter_fn(data)
    if df_wrapper.is_column(condition):
        result = []
        for idx, mask in enumerate(df_wrapper.iterate(condition)):
            if mask:
                result.append(df_wrapper.get_item(data, idx))
        if len(result):
            return (True, df_wrapper.concat(result))
        else:
            return (False, None)
    if not isinstance(condition, bool):
        raise ValueError('Boolean output is required for `filter_fn` of FilterIterDataPipe, got', type(condition))
    return (condition, data)