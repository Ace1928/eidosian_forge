from __future__ import annotations
import operator
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import base
from .collections import collection
from .collections import collection_adapter
from .. import exc as sa_exc
from .. import util
from ..sql import coercions
from ..sql import expression
from ..sql import roles
from ..util.typing import Literal
@classmethod
def _unreduce(cls, keyfunc: _F, values: Dict[_KT, _KT], adapter: Optional[CollectionAdapter]=None) -> 'KeyFuncDict[_KT, _KT]':
    mp: KeyFuncDict[_KT, _KT] = KeyFuncDict(keyfunc)
    mp.update(values)
    return mp