from __future__ import annotations
from typing import Any, Mapping, Optional, cast
from pymongo.errors import InvalidOperation
class DeleteResult(_WriteResult):
    """The return type for :meth:`~pymongo.collection.Collection.delete_one`
    and :meth:`~pymongo.collection.Collection.delete_many`
    """
    __slots__ = ('__raw_result',)

    def __init__(self, raw_result: Mapping[str, Any], acknowledged: bool) -> None:
        self.__raw_result = raw_result
        super().__init__(acknowledged)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.__raw_result!r}, acknowledged={self.acknowledged})'

    @property
    def raw_result(self) -> Mapping[str, Any]:
        """The raw result document returned by the server."""
        return self.__raw_result

    @property
    def deleted_count(self) -> int:
        """The number of documents deleted."""
        self._raise_if_unacknowledged('deleted_count')
        return self.__raw_result.get('n', 0)