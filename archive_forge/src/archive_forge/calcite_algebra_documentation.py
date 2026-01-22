import abc
from .dataframe.utils import ColNameCodec
from .db_worker import DbTable
from .expr import BaseExpr

        Reset ID to be used for the next new node to `next_id`.

        Can be used to have a zero-based numbering for each
        generated query.

        Parameters
        ----------
        next_id : int, default: 0
            Next node id.
        