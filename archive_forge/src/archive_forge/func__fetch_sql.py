from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Sequence
from lazyops.libs.dbinit.base import  Engine, Row, TextClause
@staticmethod
def _fetch_sql(engine: Engine, statement: TextClause) -> Sequence[Row[Any]]:
    """
        Fetches results from the database specified with the provided engine.
        :param engine: A :class:`sqlalchemy.Engine` for the target database.
        :param statement: A single :class:`sqlalchemy.TextClause` statement to fetch information from the database.
        :return: A Sequence of :class:`sqlalchemy.Row` that contain the results of the query provided.
        """
    with engine.connect() as conn:
        result = conn.execute(statement)
        return result.all()