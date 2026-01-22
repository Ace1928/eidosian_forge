from __future__ import annotations as _annotations
from collections import deque
from dataclasses import dataclass, field
from functools import cached_property, partial, partialmethod
from inspect import Parameter, Signature, isdatadescriptor, ismethoddescriptor, signature
from itertools import islice
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Generic, Iterable, TypeVar, Union
from pydantic_core import PydanticUndefined, core_schema
from typing_extensions import Literal, TypeAlias, is_typeddict
from ..errors import PydanticUserError
from ._core_utils import get_type_ref
from ._internal_dataclass import slots_true
from ._typing_extra import get_function_type_hints
def _serializer_info_arg(mode: Literal['plain', 'wrap'], n_positional: int) -> bool | None:
    if mode == 'plain':
        if n_positional == 1:
            return False
        elif n_positional == 2:
            return True
    else:
        assert mode == 'wrap', f"invalid mode: {mode!r}, expected 'plain' or 'wrap'"
        if n_positional == 2:
            return False
        elif n_positional == 3:
            return True
    return None