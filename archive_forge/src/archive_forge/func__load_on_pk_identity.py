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
def _load_on_pk_identity(self, session, query, primary_key_identity, **kw):
    """Load the given primary key identity from the database."""
    mapper = query._raw_columns[0]._annotations['parententity']
    _get_clause, _get_params = mapper._get_clause

    def setup(query):
        _lcl_get_clause = _get_clause
        q = query._clone()
        q._get_condition()
        q._order_by = None
        if None in primary_key_identity:
            nones = {_get_params[col].key for col, value in zip(mapper.primary_key, primary_key_identity) if value is None}
            _lcl_get_clause = sql_util.adapt_criterion_to_null(_lcl_get_clause, nones)
        q._where_criteria = (sql_util._deep_annotate(_lcl_get_clause, {'_orm_adapt': True}),)
        for fn in self._post_criteria:
            q = fn(q)
        return q
    bq = self.bq
    bq = bq._clone()
    bq._cache_key += (_get_clause,)
    bq = bq.with_criteria(setup, tuple((elem is None for elem in primary_key_identity)))
    params = {_get_params[primary_key].key: id_val for id_val, primary_key in zip(primary_key_identity, mapper.primary_key)}
    result = list(bq.for_session(self.session).params(**params))
    l = len(result)
    if l > 1:
        raise orm_exc.MultipleResultsFound()
    elif l:
        return result[0]
    else:
        return None