from typing import Union
from .aggregation import Asc, Desc, Reducer, SortDirection
class tolist(FieldOnlyReducer):
    """
    Returns all the matched properties in a list
    """
    NAME = 'TOLIST'

    def __init__(self, field: str) -> None:
        super().__init__(field)