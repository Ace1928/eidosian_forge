from typing import Union
from .aggregation import Asc, Desc, Reducer, SortDirection
class avg(FieldOnlyReducer):
    """
    Calculates the mean value in the given field within the group
    """
    NAME = 'AVG'

    def __init__(self, field: str) -> None:
        super().__init__(field)