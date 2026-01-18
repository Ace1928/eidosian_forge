from __future__ import annotations
import collections
import types
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Deque
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union
import weakref
from .. import exc
from .. import util
def with_dispatch_target(self, dispatch_target: Any) -> _EventKey[_ET]:
    if dispatch_target is self.dispatch_target:
        return self
    else:
        return _EventKey(self.target, self.identifier, self.fn, dispatch_target, _fn_wrap=self.fn_wrap)