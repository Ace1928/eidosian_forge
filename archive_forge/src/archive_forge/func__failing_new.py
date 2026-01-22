import inspect
import operator
import typing as t
from collections import deque
from markupsafe import Markup
from .utils import _PassArg
def _failing_new(*args: t.Any, **kwargs: t.Any) -> 'te.NoReturn':
    raise TypeError("can't create custom node types")