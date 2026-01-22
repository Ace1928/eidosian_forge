from typing import Union
from .aggregation import Asc, Desc, Reducer, SortDirection
class FieldOnlyReducer(Reducer):
    """See https://redis.io/docs/interact/search-and-query/search/aggregations/"""

    def __init__(self, field: str) -> None:
        super().__init__(field)
        self._field = field