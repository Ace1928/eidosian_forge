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
def with_post_criteria(self, fn):
    """Add a criteria function that will be applied post-cache.

        This adds a function that will be run against the
        :class:`_query.Query` object after it is retrieved from the
        cache.    This currently includes **only** the
        :meth:`_query.Query.params` and :meth:`_query.Query.execution_options`
        methods.

        .. warning::  :meth:`_baked.Result.with_post_criteria`
           functions are applied
           to the :class:`_query.Query`
           object **after** the query's SQL statement
           object has been retrieved from the cache.   Only
           :meth:`_query.Query.params` and
           :meth:`_query.Query.execution_options`
           methods should be used.


        .. versionadded:: 1.2


        """
    return self._using_post_criteria([fn])