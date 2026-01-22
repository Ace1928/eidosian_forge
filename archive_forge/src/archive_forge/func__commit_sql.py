from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Sequence
from lazyops.libs.dbinit.base import  Engine, Row, TextClause
@staticmethod
def _commit_sql(engine: Engine, statements: Sequence[TextClause]) -> None:
    """
        Commits SQL statements to the database specified with the provided engine.
        :param engine: A :class:`sqlalchemy.Engine` for the target database.
        :param statements: A Sequence of :class:`sqlalchemy.TextClause` statements to commit.
        """
    with engine.connect() as conn:
        for statement in statements:
            conn.execution_options(isolation_level='AUTOCOMMIT').execute(statement)
            if hasattr(conn, 'commit'):
                conn.commit()