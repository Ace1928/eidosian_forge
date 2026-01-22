import functools
from collections import namedtuple
from typing import Callable, Iterator, Sized, TypeVar, Optional, Union, Any, Dict, List
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.datapipes.dataframe import dataframe_wrapper as df_wrapper
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.utils.common import (_check_unpickable_fn,
@functional_datapipe('map')
class MapperIterDataPipe(IterDataPipe[T_co]):
    """
    Applies a function over each item from the source DataPipe (functional name: ``map``).

    The function can be any regular Python function or partial object. Lambda
    function is not recommended as it is not supported by pickle.

    Args:
        datapipe: Source Iterable DataPipe
        fn: Function being applied over each item
        input_col: Index or indices of data which ``fn`` is applied, such as:

            - ``None`` as default to apply ``fn`` to the data directly.
            - Integer(s) is used for list/tuple.
            - Key(s) is used for dict.

        output_col: Index of data where result of ``fn`` is placed. ``output_col`` can be specified
            only when ``input_col`` is not ``None``

            - ``None`` as default to replace the index that ``input_col`` specified; For ``input_col`` with
              multiple indices, the left-most one is used, and other indices will be removed.
            - Integer is used for list/tuple. ``-1`` represents to append result at the end.
            - Key is used for dict. New key is acceptable.

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import IterableWrapper, Mapper
        >>> def add_one(x):
        ...     return x + 1
        >>> dp = IterableWrapper(range(10))
        >>> map_dp_1 = dp.map(add_one)  # Invocation via functional form is preferred
        >>> list(map_dp_1)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> # We discourage the usage of `lambda` functions as they are not serializable with `pickle`
        >>> # Use `functools.partial` or explicitly define the function instead
        >>> map_dp_2 = Mapper(dp, lambda x: x + 1)
        >>> list(map_dp_2)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    """
    datapipe: IterDataPipe
    fn: Callable

    def __init__(self, datapipe: IterDataPipe, fn: Callable, input_col=None, output_col=None) -> None:
        super().__init__()
        self.datapipe = datapipe
        _check_unpickable_fn(fn)
        self.fn = fn
        self.input_col = input_col
        if input_col is None and output_col is not None:
            raise ValueError('`output_col` must be None when `input_col` is None.')
        if isinstance(output_col, (list, tuple)):
            if len(output_col) > 1:
                raise ValueError('`output_col` must be a single-element list or tuple')
            output_col = output_col[0]
        self.output_col = output_col
        validate_input_col(fn, input_col)

    def _apply_fn(self, data):
        if self.input_col is None and self.output_col is None:
            return self.fn(data)
        if self.input_col is None:
            res = self.fn(data)
        elif isinstance(self.input_col, (list, tuple)):
            args = tuple((data[col] for col in self.input_col))
            res = self.fn(*args)
        else:
            res = self.fn(data[self.input_col])
        if isinstance(data, tuple):
            t_flag = True
            data = list(data)
        else:
            t_flag = False
        if self.output_col is None:
            if isinstance(self.input_col, (list, tuple)):
                data[self.input_col[0]] = res
                for idx in sorted(self.input_col[1:], reverse=True):
                    del data[idx]
            else:
                data[self.input_col] = res
        elif self.output_col == -1:
            data.append(res)
        else:
            data[self.output_col] = res
        return tuple(data) if t_flag else data

    def __iter__(self) -> Iterator[T_co]:
        for data in self.datapipe:
            yield self._apply_fn(data)

    def __len__(self) -> int:
        if isinstance(self.datapipe, Sized):
            return len(self.datapipe)
        raise TypeError(f"{type(self).__name__} instance doesn't have valid length")