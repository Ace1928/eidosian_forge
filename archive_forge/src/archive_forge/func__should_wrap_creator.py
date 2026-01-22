from __future__ import annotations
from collections import deque
import dataclasses
from enum import Enum
import threading
import time
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Deque
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
import weakref
from .. import event
from .. import exc
from .. import log
from .. import util
from ..util.typing import Literal
from ..util.typing import Protocol
def _should_wrap_creator(self, creator: Union[_CreatorFnType, _CreatorWRecFnType]) -> _CreatorWRecFnType:
    """Detect if creator accepts a single argument, or is sent
        as a legacy style no-arg function.

        """
    try:
        argspec = util.get_callable_argspec(self._creator, no_self=True)
    except TypeError:
        creator_fn = cast(_CreatorFnType, creator)
        return lambda rec: creator_fn()
    if argspec.defaults is not None:
        defaulted = len(argspec.defaults)
    else:
        defaulted = 0
    positionals = len(argspec[0]) - defaulted
    if (argspec[0], argspec[3]) == (['connection_record'], (None,)):
        return cast(_CreatorWRecFnType, creator)
    elif positionals == 1:
        return cast(_CreatorWRecFnType, creator)
    else:
        creator_fn = cast(_CreatorFnType, creator)
        return lambda rec: creator_fn()