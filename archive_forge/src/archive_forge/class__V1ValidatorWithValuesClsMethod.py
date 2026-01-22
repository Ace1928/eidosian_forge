from __future__ import annotations as _annotations
from functools import partial, partialmethod
from types import FunctionType
from typing import TYPE_CHECKING, Any, Callable, TypeVar, Union, overload
from warnings import warn
from typing_extensions import Literal, Protocol, TypeAlias
from .._internal import _decorators, _decorators_v1
from ..errors import PydanticUserError
from ..warnings import PydanticDeprecatedSince20
class _V1ValidatorWithValuesClsMethod(Protocol):

    def __call__(self, __cls: Any, __value: Any, values: dict[str, Any]) -> Any:
        ...