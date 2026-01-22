import collections.abc as collections_abc
import logging
from .. import exc as sa_exc
from .. import util
from ..orm import exc as orm_exc
from ..orm.query import Query
from ..orm.session import Session
from ..sql import func
from ..sql import literal_column
from ..sql import util as sql_util
def _effective_key(self, session):
    """Return the key that actually goes into the cache dictionary for
        this :class:`.BakedQuery`, taking into account the given
        :class:`.Session`.

        This basically means we also will include the session's query_class,
        as the actual :class:`_query.Query` object is part of what's cached
        and needs to match the type of :class:`_query.Query` that a later
        session will want to use.

        """
    return self._cache_key + (session._query_cls,)