from __future__ import annotations
import inspect
import sys
import warnings
from types import ModuleType
from typing import AbstractSet
from typing import Any
from typing import Callable
from typing import Final
from typing import final
from typing import Generator
from typing import List
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypedDict
from typing import TypeVar
from typing import Union
from ._result import Result
def _add_hookimpl(self, hookimpl: HookImpl) -> None:
    """Add an implementation to the callback chain."""
    for i, method in enumerate(self._hookimpls):
        if method.hookwrapper or method.wrapper:
            splitpoint = i
            break
    else:
        splitpoint = len(self._hookimpls)
    if hookimpl.hookwrapper or hookimpl.wrapper:
        start, end = (splitpoint, len(self._hookimpls))
    else:
        start, end = (0, splitpoint)
    if hookimpl.trylast:
        self._hookimpls.insert(start, hookimpl)
    elif hookimpl.tryfirst:
        self._hookimpls.insert(end, hookimpl)
    else:
        i = end - 1
        while i >= start and self._hookimpls[i].tryfirst:
            i -= 1
        self._hookimpls.insert(i + 1, hookimpl)