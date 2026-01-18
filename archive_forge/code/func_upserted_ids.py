from __future__ import annotations
from typing import Any, Mapping, Optional, cast
from pymongo.errors import InvalidOperation
@property
def upserted_ids(self) -> Optional[dict[int, Any]]:
    """A map of operation index to the _id of the upserted document."""
    self._raise_if_unacknowledged('upserted_ids')
    if self.__bulk_api_result:
        return {upsert['index']: upsert['_id'] for upsert in self.bulk_api_result['upserted']}
    return None