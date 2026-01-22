from typing import Any, Mapping, Type, Union
import torch
from torch import Tensor
class _DtypeContextManager:
    """A context manager to change the default tensor type when tensors get created.

    See: :func:`torch.set_default_dtype`

    """

    def __init__(self, dtype: torch.dtype) -> None:
        self._previous_dtype: torch.dtype = torch.get_default_dtype()
        self._new_dtype = dtype

    def __enter__(self) -> None:
        torch.set_default_dtype(self._new_dtype)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        torch.set_default_dtype(self._previous_dtype)