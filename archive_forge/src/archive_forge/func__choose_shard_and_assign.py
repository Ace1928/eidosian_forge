from __future__ import annotations
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .. import event
from .. import exc
from .. import inspect
from .. import util
from ..orm import PassiveFlag
from ..orm._typing import OrmExecuteOptionsParameter
from ..orm.interfaces import ORMOption
from ..orm.mapper import Mapper
from ..orm.query import Query
from ..orm.session import _BindArguments
from ..orm.session import _PKIdentityArgument
from ..orm.session import Session
from ..util.typing import Protocol
from ..util.typing import Self
def _choose_shard_and_assign(self, mapper: Optional[_EntityBindKey[_O]], instance: Any, **kw: Any) -> Any:
    if instance is not None:
        state = inspect(instance)
        if state.key:
            token = state.key[2]
            assert token is not None
            return token
        elif state.identity_token:
            return state.identity_token
    assert isinstance(mapper, Mapper)
    shard_id = self.shard_chooser(mapper, instance, **kw)
    if instance is not None:
        state.identity_token = shard_id
    return shard_id