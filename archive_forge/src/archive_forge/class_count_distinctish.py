from typing import Union
from .aggregation import Asc, Desc, Reducer, SortDirection
class count_distinctish(FieldOnlyReducer):
    """
    Calculate the number of distinct values contained in all the results in the
    group for the given field. This uses a faster algorithm than
    `count_distinct` but is less accurate
    """
    NAME = 'COUNT_DISTINCTISH'