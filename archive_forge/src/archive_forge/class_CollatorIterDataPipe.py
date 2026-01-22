import functools
from collections import namedtuple
from typing import Callable, Iterator, Sized, TypeVar, Optional, Union, Any, Dict, List
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.datapipes.dataframe import dataframe_wrapper as df_wrapper
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.utils.common import (_check_unpickable_fn,
@functional_datapipe('collate')
class CollatorIterDataPipe(MapperIterDataPipe):
    """
    Collates samples from DataPipe to Tensor(s) by a custom collate function (functional name: ``collate``).

    By default, it uses :func:`torch.utils.data.default_collate`.

    .. note::
        While writing a custom collate function, you can import :func:`torch.utils.data.default_collate` for the
        default behavior and `functools.partial` to specify any additional arguments.

    Args:
        datapipe: Iterable DataPipe being collated
        collate_fn: Customized collate function to collect and combine data or a batch of data.
            Default function collates to Tensor(s) based on data type.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Convert integer data to float Tensor
        >>> class MyIterDataPipe(torch.utils.data.IterDataPipe):
        ...     def __init__(self, start, end):
        ...         super(MyIterDataPipe).__init__()
        ...         assert end > start, "this example code only works with end >= start"
        ...         self.start = start
        ...         self.end = end
        ...
        ...     def __iter__(self):
        ...         return iter(range(self.start, self.end))
        ...
        ...     def __len__(self):
        ...         return self.end - self.start
        ...
        >>> ds = MyIterDataPipe(start=3, end=7)
        >>> print(list(ds))
        [3, 4, 5, 6]
        >>> def collate_fn(batch):
        ...     return torch.tensor(batch, dtype=torch.float)
        ...
        >>> collated_ds = CollateIterDataPipe(ds, collate_fn=collate_fn)
        >>> print(list(collated_ds))
        [tensor(3.), tensor(4.), tensor(5.), tensor(6.)]
    """

    def __init__(self, datapipe: IterDataPipe, conversion: Optional[Union[Callable[..., Any], Dict[Union[str, Any], Union[Callable, Any]]]]=default_collate, collate_fn: Optional[Callable]=None) -> None:
        if collate_fn is not None:
            super().__init__(datapipe, fn=collate_fn)
        elif callable(conversion):
            super().__init__(datapipe, fn=conversion)
        else:
            collate_fn = functools.partial(_collate_helper, conversion)
            super().__init__(datapipe, fn=collate_fn)