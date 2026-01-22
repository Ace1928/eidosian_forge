from typing import Union
from .aggregation import Asc, Desc, Reducer, SortDirection
class count(Reducer):
    """
    Counts the number of results in the group
    """
    NAME = 'COUNT'

    def __init__(self) -> None:
        super().__init__()