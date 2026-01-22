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
class _MKeyfuncMapped(KeyFuncDict[_KT, _KT]):

    def __init__(self, *dict_args: Any) -> None:
        super().__init__(keyfunc, *dict_args, ignore_unpopulated_attribute=ignore_unpopulated_attribute)