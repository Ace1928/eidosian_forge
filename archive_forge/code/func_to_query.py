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
def to_query(self, query_or_session):
    """Return the :class:`_query.Query` object for use as a subquery.

        This method should be used within the lambda callable being used
        to generate a step of an enclosing :class:`.BakedQuery`.   The
        parameter should normally be the :class:`_query.Query` object that
        is passed to the lambda::

            sub_bq = self.bakery(lambda s: s.query(User.name))
            sub_bq += lambda q: q.filter(
                User.id == Address.user_id).correlate(Address)

            main_bq = self.bakery(lambda s: s.query(Address))
            main_bq += lambda q: q.filter(
                sub_bq.to_query(q).exists())

        In the case where the subquery is used in the first callable against
        a :class:`.Session`, the :class:`.Session` is also accepted::

            sub_bq = self.bakery(lambda s: s.query(User.name))
            sub_bq += lambda q: q.filter(
                User.id == Address.user_id).correlate(Address)

            main_bq = self.bakery(
                lambda s: s.query(
                Address.id, sub_bq.to_query(q).scalar_subquery())
            )

        :param query_or_session: a :class:`_query.Query` object or a class
         :class:`.Session` object, that is assumed to be within the context
         of an enclosing :class:`.BakedQuery` callable.


         .. versionadded:: 1.3


        """
    if isinstance(query_or_session, Session):
        session = query_or_session
    elif isinstance(query_or_session, Query):
        session = query_or_session.session
        if session is None:
            raise sa_exc.ArgumentError('Given Query needs to be associated with a Session')
    else:
        raise TypeError('Query or Session object expected, got %r.' % type(query_or_session))
    return self._as_query(session)