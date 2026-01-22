from __future__ import annotations as _annotations
from functools import partial, partialmethod
from types import FunctionType
from typing import TYPE_CHECKING, Any, Callable, TypeVar, Union, overload
from warnings import warn
from typing_extensions import Literal, Protocol, TypeAlias
from .._internal import _decorators, _decorators_v1
from ..errors import PydanticUserError
from ..warnings import PydanticDeprecatedSince20
Decorate methods on a model indicating that they should be used to validate (and perhaps
    modify) data either before or after standard model parsing/validation is performed.

    Args:
        pre (bool, optional): Whether this validator should be called before the standard
            validators (else after). Defaults to False.
        skip_on_failure (bool, optional): Whether to stop validation and return as soon as a
            failure is encountered. Defaults to False.
        allow_reuse (bool, optional): Whether to track and raise an error if another validator
            refers to the decorated function. Defaults to False.

    Returns:
        Any: A decorator that can be used to decorate a function to be used as a root_validator.
    