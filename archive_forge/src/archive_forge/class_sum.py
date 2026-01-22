from typing import Union
from .aggregation import Asc, Desc, Reducer, SortDirection
class sum(FieldOnlyReducer):
    """
    Calculates the sum of all the values in the given fields within the group
    """
    NAME = 'SUM'

    def __init__(self, field: str) -> None:
        super().__init__(field)