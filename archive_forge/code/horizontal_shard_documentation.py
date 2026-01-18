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
Construct a :class:`_horizontal.set_shard_id` option.

        :param shard_id: shard identifier
        :param propagate_to_loaders: if left at its default of ``True``, the
         shard option will take place for lazy loaders such as
         :func:`_orm.lazyload` and :func:`_orm.defer`; if False, the option
         will not be propagated to loaded objects. Note that :func:`_orm.defer`
         always limits to the shard_id of the parent row in any case, so the
         parameter only has a net effect on the behavior of the
         :func:`_orm.lazyload` strategy.

        