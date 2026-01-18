from typing import Callable, Iterator, Tuple, TypeVar
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.dataframe import dataframe_wrapper as df_wrapper
from torch.utils.data.datapipes.utils.common import (

    Filters out elements from the source datapipe according to input ``filter_fn`` (functional name: ``filter``).

    Args:
        datapipe: Iterable DataPipe being filtered
        filter_fn: Customized function mapping an element to a boolean.
        input_col: Index or indices of data which ``filter_fn`` is applied, such as:

            - ``None`` as default to apply ``filter_fn`` to the data directly.
            - Integer(s) is used for list/tuple.
            - Key(s) is used for dict.

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> def is_even(n):
        ...     return n % 2 == 0
        >>> dp = IterableWrapper(range(5))
        >>> filter_dp = dp.filter(filter_fn=is_even)
        >>> list(filter_dp)
        [0, 2, 4]
    