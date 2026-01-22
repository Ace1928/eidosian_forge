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
class set_shard_id(ORMOption):
    """a loader option for statements to apply a specific shard id to the
    primary query as well as for additional relationship and column
    loaders.

    The :class:`_horizontal.set_shard_id` option may be applied using
    the :meth:`_sql.Executable.options` method of any executable statement::

        stmt = (
            select(MyObject).
            where(MyObject.name == 'some name').
            options(set_shard_id("shard1"))
        )

    Above, the statement when invoked will limit to the "shard1" shard
    identifier for the primary query as well as for all relationship and
    column loading strategies, including eager loaders such as
    :func:`_orm.selectinload`, deferred column loaders like :func:`_orm.defer`,
    and the lazy relationship loader :func:`_orm.lazyload`.

    In this way, the :class:`_horizontal.set_shard_id` option has much wider
    scope than using the "shard_id" argument within the
    :paramref:`_orm.Session.execute.bind_arguments` dictionary.


    .. versionadded:: 2.0.0

    """
    __slots__ = ('shard_id', 'propagate_to_loaders')

    def __init__(self, shard_id: ShardIdentifier, propagate_to_loaders: bool=True):
        """Construct a :class:`_horizontal.set_shard_id` option.

        :param shard_id: shard identifier
        :param propagate_to_loaders: if left at its default of ``True``, the
         shard option will take place for lazy loaders such as
         :func:`_orm.lazyload` and :func:`_orm.defer`; if False, the option
         will not be propagated to loaded objects. Note that :func:`_orm.defer`
         always limits to the shard_id of the parent row in any case, so the
         parameter only has a net effect on the behavior of the
         :func:`_orm.lazyload` strategy.

        """
        self.shard_id = shard_id
        self.propagate_to_loaders = propagate_to_loaders