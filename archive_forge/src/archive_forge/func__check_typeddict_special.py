import sys
import typing
from collections.abc import Callable
from os import PathLike
from typing import (  # type: ignore
from typing_extensions import (
def _check_typeddict_special(type_: Any) -> bool:
    return type_ is TypedDictRequired or type_ is TypedDictNotRequired