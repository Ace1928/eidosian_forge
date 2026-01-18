from __future__ import annotations
from typing import Any, Mapping, Optional, cast
from pymongo.errors import InvalidOperation
@property
def upserted_id(self) -> Any:
    """The _id of the inserted document if an upsert took place. Otherwise
        ``None``.
        """
    self._raise_if_unacknowledged('upserted_id')
    assert self.__raw_result is not None
    return self.__raw_result.get('upserted')