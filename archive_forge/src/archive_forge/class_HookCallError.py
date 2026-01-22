from __future__ import annotations
from types import TracebackType
from typing import Callable
from typing import cast
from typing import final
from typing import Generic
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TypeVar
class HookCallError(Exception):
    """Hook was called incorrectly."""