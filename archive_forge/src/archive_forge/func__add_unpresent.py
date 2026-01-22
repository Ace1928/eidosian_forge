from __future__ import annotations
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import NoReturn
from typing import Optional
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
import weakref
from . import util as orm_util
from .. import exc as sa_exc
def _add_unpresent(self, state: InstanceState[Any], key: _IdentityKeyType[Any]) -> None:
    self._dict[key] = state
    state._instance_dict = self._wr