from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Sequence
from lazyops.libs.dbinit.base import  Engine, Row, TextClause
class SQLCreatable(SQLBase):

    @abstractmethod
    def _create(self) -> None:
        """
        Create this entity in the cluster.
        """
        pass

    @abstractmethod
    def _exists(self) -> bool:
        """
        Check if this entity currently exists in the cluster.
        :return: True if it exists, False if it does not.
        """
        pass

    @abstractmethod
    def _drop(self) -> None:
        """
        Drop this entity from the cluster.
        """
        pass

    @abstractmethod
    def _create_statements(self) -> Sequence[TextClause]:
        """
        The SQL statements that create this entity.
        :return: A Sequence of :class:`sqlalchemy.TextClause` containing the SQL to create this entity.
        """
        pass

    @abstractmethod
    def _exists_statement(self) -> TextClause:
        """
        The SQL statement that checks to see if this entity exists.
        :return: A single :class:`sqlalchemy.TextClause` containing the SQL to check if this entity exists.
        """
        pass

    @abstractmethod
    def _drop_statements(self) -> Sequence[TextClause]:
        """
        The SQL statements that drop this entity.
        :return: A Sequence of :class:`sqlalchemy.TextClause` containing the SQL to drop this entity.
        """
        pass