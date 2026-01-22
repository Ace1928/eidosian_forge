from __future__ import annotations
from typing import Any, Mapping, Optional, cast
from pymongo.errors import InvalidOperation
class BulkWriteResult(_WriteResult):
    """An object wrapper for bulk API write results."""
    __slots__ = ('__bulk_api_result',)

    def __init__(self, bulk_api_result: dict[str, Any], acknowledged: bool) -> None:
        """Create a BulkWriteResult instance.

        :Parameters:
          - `bulk_api_result`: A result dict from the bulk API
          - `acknowledged`: Was this write result acknowledged? If ``False``
            then all properties of this object will raise
            :exc:`~pymongo.errors.InvalidOperation`.
        """
        self.__bulk_api_result = bulk_api_result
        super().__init__(acknowledged)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.__bulk_api_result!r}, acknowledged={self.acknowledged})'

    @property
    def bulk_api_result(self) -> dict[str, Any]:
        """The raw bulk API result."""
        return self.__bulk_api_result

    @property
    def inserted_count(self) -> int:
        """The number of documents inserted."""
        self._raise_if_unacknowledged('inserted_count')
        return cast(int, self.__bulk_api_result.get('nInserted'))

    @property
    def matched_count(self) -> int:
        """The number of documents matched for an update."""
        self._raise_if_unacknowledged('matched_count')
        return cast(int, self.__bulk_api_result.get('nMatched'))

    @property
    def modified_count(self) -> int:
        """The number of documents modified."""
        self._raise_if_unacknowledged('modified_count')
        return cast(int, self.__bulk_api_result.get('nModified'))

    @property
    def deleted_count(self) -> int:
        """The number of documents deleted."""
        self._raise_if_unacknowledged('deleted_count')
        return cast(int, self.__bulk_api_result.get('nRemoved'))

    @property
    def upserted_count(self) -> int:
        """The number of documents upserted."""
        self._raise_if_unacknowledged('upserted_count')
        return cast(int, self.__bulk_api_result.get('nUpserted'))

    @property
    def upserted_ids(self) -> Optional[dict[int, Any]]:
        """A map of operation index to the _id of the upserted document."""
        self._raise_if_unacknowledged('upserted_ids')
        if self.__bulk_api_result:
            return {upsert['index']: upsert['_id'] for upsert in self.bulk_api_result['upserted']}
        return None