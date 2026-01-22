import ast
import dataclasses
import inspect
from inspect import CO_VARARGS
from inspect import CO_VARKEYWORDS
from io import StringIO
import os
from pathlib import Path
import re
import sys
import traceback
from traceback import format_exception_only
from types import CodeType
from types import FrameType
from types import TracebackType
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Dict
from typing import Final
from typing import final
from typing import Generic
from typing import Iterable
from typing import List
from typing import Literal
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Pattern
from typing import Sequence
from typing import Set
from typing import SupportsIndex
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
import pluggy
import _pytest
from _pytest._code.source import findsource
from _pytest._code.source import getrawcode
from _pytest._code.source import getstatementrange_ast
from _pytest._code.source import Source
from _pytest._io import TerminalWriter
from _pytest._io.saferepr import safeformat
from _pytest._io.saferepr import saferepr
from _pytest.compat import get_real_func
from _pytest.deprecated import check_ispytest
from _pytest.pathlib import absolutepath
from _pytest.pathlib import bestrelpath
def _group_contains(self, exc_group: BaseExceptionGroup[BaseException], expected_exception: Union[Type[BaseException], Tuple[Type[BaseException], ...]], match: Union[str, Pattern[str], None], target_depth: Optional[int]=None, current_depth: int=1) -> bool:
    """Return `True` if a `BaseExceptionGroup` contains a matching exception."""
    if target_depth is not None and current_depth > target_depth:
        return False
    for exc in exc_group.exceptions:
        if isinstance(exc, BaseExceptionGroup):
            if self._group_contains(exc, expected_exception, match, target_depth, current_depth + 1):
                return True
        if target_depth is not None and current_depth != target_depth:
            continue
        if not isinstance(exc, expected_exception):
            continue
        if match is not None:
            value = self._stringify_exception(exc)
            if not re.search(match, value):
                continue
        return True
    return False