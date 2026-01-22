from __future__ import annotations
import typing
from typing import Any
from typing import cast
from typing import Dict
from typing import Generic
from typing import Iterator
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type
from typing import Union
import weakref
from .attr import _ClsLevelDispatch
from .attr import _EmptyListener
from .attr import _InstanceLevelDispatch
from .attr import _JoinedListener
from .registry import _ET
from .registry import _EventKey
from .. import util
from ..util.typing import Literal
class slots_dispatcher(dispatcher[_ET]):

    def __get__(self, obj: Any, cls: Type[Any]) -> Any:
        if obj is None:
            return self.dispatch
        if hasattr(obj, '_slots_dispatch'):
            return obj._slots_dispatch
        disp = self.dispatch._for_instance(obj)
        obj._slots_dispatch = disp
        return disp