import warnings
from collections import OrderedDict, abc as container_abcs
from itertools import chain, islice
import operator
import torch
from .module import Module
from ..parameter import Parameter
from torch._jit_internal import _copy_to_script_wrapper
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, overload, Tuple, TypeVar, Union
from typing_extensions import Self
class ParameterList(Module):
    """Holds parameters in a list.

    :class:`~torch.nn.ParameterList` can be used like a regular Python
    list, but Tensors that are :class:`~torch.nn.Parameter` are properly registered,
    and will be visible by all :class:`~torch.nn.Module` methods.

    Note that the constructor, assigning an element of the list, the
    :meth:`~torch.nn.ParameterDict.append` method and the :meth:`~torch.nn.ParameterDict.extend`
    method will convert any :class:`~torch.Tensor` into :class:`~torch.nn.Parameter`.

    Args:
        parameters (iterable, optional): an iterable of elements to add to the list.

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.params = nn.ParameterList([nn.Parameter(torch.randn(10, 10)) for i in range(10)])

            def forward(self, x):
                # ParameterList can act as an iterable, or be indexed using ints
                for i, p in enumerate(self.params):
                    x = self.params[i // 2].mm(x) + p.mm(x)
                return x
    """

    def __init__(self, values: Optional[Iterable[Any]]=None) -> None:
        super().__init__()
        self._size = 0
        if values is not None:
            self += values

    def _get_abs_string_index(self, idx):
        """Get the absolute index for the list of modules."""
        idx = operator.index(idx)
        if not -len(self) <= idx < len(self):
            raise IndexError(f'index {idx} is out of range')
        if idx < 0:
            idx += len(self)
        return str(idx)

    @overload
    def __getitem__(self, idx: int) -> Any:
        ...

    @overload
    def __getitem__(self: T, idx: slice) -> T:
        ...

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            out = self.__class__()
            for i in range(start, stop, step):
                out.append(self[i])
            return out
        else:
            idx = self._get_abs_string_index(idx)
            return getattr(self, str(idx))

    def __setitem__(self, idx: int, param: Any) -> None:
        idx = self._get_abs_string_index(idx)
        if isinstance(param, torch.Tensor) and (not isinstance(param, Parameter)):
            param = Parameter(param)
        return setattr(self, str(idx), param)

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[Any]:
        return iter((self[i] for i in range(len(self))))

    def __iadd__(self, parameters: Iterable[Any]) -> Self:
        return self.extend(parameters)

    def __dir__(self):
        keys = super().__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def append(self, value: Any) -> 'ParameterList':
        """Append a given value at the end of the list.

        Args:
            value (Any): value to append
        """
        new_idx = len(self)
        self._size += 1
        self[new_idx] = value
        return self

    def extend(self, values: Iterable[Any]) -> Self:
        """Append values from a Python iterable to the end of the list.

        Args:
            values (iterable): iterable of values to append
        """
        if not isinstance(values, container_abcs.Iterable) or isinstance(values, torch.Tensor):
            raise TypeError('ParameterList.extend should be called with an iterable, but got ' + type(values).__name__)
        for value in values:
            self.append(value)
        return self

    def extra_repr(self) -> str:
        child_lines = []
        for k, p in enumerate(self):
            if isinstance(p, torch.Tensor):
                size_str = 'x'.join((str(size) for size in p.size()))
                if p.device.type in ['cuda', torch._C._get_privateuse1_backend_name()]:
                    device_str = f' ({p.device})'
                else:
                    device_str = ''
                parastr = '{} containing: [{} of size {}{}]'.format('Parameter' if isinstance(p, Parameter) else 'Tensor', p.dtype, size_str, device_str)
                child_lines.append('  (' + str(k) + '): ' + parastr)
            else:
                child_lines.append('  (' + str(k) + '): Object of type: ' + type(p).__name__)
        tmpstr = '\n'.join(child_lines)
        return tmpstr

    def __call__(self, *args, **kwargs):
        raise RuntimeError('ParameterList should not be called.')