from __future__ import annotations
import asyncio
from contextvars import Context
import sys
import typing
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import Coroutine
from typing import Optional
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .langhelpers import memoized_property
from .. import exc
from ..util import py311
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import Self
from ..util.typing import TypeGuard
Return embedded event loop.