from __future__ import annotations
import collections.abc as collections_abc
import operator
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import attributes
from . import interfaces
from . import loading
from . import util as orm_util
from ._typing import _O
from .base import _assertions
from .context import _column_descriptions
from .context import _determine_last_joined_entity
from .context import _legacy_filter_by_entity_zero
from .context import FromStatement
from .context import ORMCompileState
from .context import QueryContext
from .interfaces import ORMColumnDescription
from .interfaces import ORMColumnsClauseRole
from .util import AliasedClass
from .util import object_mapper
from .util import with_parent
from .. import exc as sa_exc
from .. import inspect
from .. import inspection
from .. import log
from .. import sql
from .. import util
from ..engine import Result
from ..engine import Row
from ..event import dispatcher
from ..event import EventTarget
from ..sql import coercions
from ..sql import expression
from ..sql import roles
from ..sql import Select
from ..sql import util as sql_util
from ..sql import visitors
from ..sql._typing import _FromClauseArgument
from ..sql._typing import _TP
from ..sql.annotation import SupportsCloneAnnotations
from ..sql.base import _entity_namespace_key
from ..sql.base import _generative
from ..sql.base import _NoArg
from ..sql.base import Executable
from ..sql.base import Generative
from ..sql.elements import BooleanClauseList
from ..sql.expression import Exists
from ..sql.selectable import _MemoizedSelectEntities
from ..sql.selectable import _SelectFromElements
from ..sql.selectable import ForUpdateArg
from ..sql.selectable import HasHints
from ..sql.selectable import HasPrefixes
from ..sql.selectable import HasSuffixes
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..sql.selectable import SelectLabelStyle
from ..util.typing import Literal
from ..util.typing import Self
@_generative
def with_for_update(self, *, nowait: bool=False, read: bool=False, of: Optional[_ForUpdateOfArgument]=None, skip_locked: bool=False, key_share: bool=False) -> Self:
    """return a new :class:`_query.Query`
        with the specified options for the
        ``FOR UPDATE`` clause.

        The behavior of this method is identical to that of
        :meth:`_expression.GenerativeSelect.with_for_update`.
        When called with no arguments,
        the resulting ``SELECT`` statement will have a ``FOR UPDATE`` clause
        appended.  When additional arguments are specified, backend-specific
        options such as ``FOR UPDATE NOWAIT`` or ``LOCK IN SHARE MODE``
        can take effect.

        E.g.::

            q = sess.query(User).populate_existing().with_for_update(nowait=True, of=User)

        The above query on a PostgreSQL backend will render like::

            SELECT users.id AS users_id FROM users FOR UPDATE OF users NOWAIT

        .. warning::

            Using ``with_for_update`` in the context of eager loading
            relationships is not officially supported or recommended by
            SQLAlchemy and may not work with certain queries on various
            database backends.  When ``with_for_update`` is successfully used
            with a query that involves :func:`_orm.joinedload`, SQLAlchemy will
            attempt to emit SQL that locks all involved tables.

        .. note::  It is generally a good idea to combine the use of the
           :meth:`_orm.Query.populate_existing` method when using the
           :meth:`_orm.Query.with_for_update` method.   The purpose of
           :meth:`_orm.Query.populate_existing` is to force all the data read
           from the SELECT to be populated into the ORM objects returned,
           even if these objects are already in the :term:`identity map`.

        .. seealso::

            :meth:`_expression.GenerativeSelect.with_for_update`
            - Core level method with
            full argument and behavioral description.

            :meth:`_orm.Query.populate_existing` - overwrites attributes of
            objects already loaded in the identity map.

        """
    self._for_update_arg = ForUpdateArg(read=read, nowait=nowait, of=of, skip_locked=skip_locked, key_share=key_share)
    return self