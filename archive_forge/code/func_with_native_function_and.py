import contextlib
import functools
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, TypeVar, Union
import torchgen.local as local
from torchgen.model import (
from torchgen.utils import context, S, T
def with_native_function_and(func: Callable[[F, F2], T]) -> Callable[[F, F2], T]:

    @functools.wraps(func)
    def wrapper(f: F, f2: F2) -> T:
        with native_function_manager(f):
            return func(f, f2)
    return wrapper