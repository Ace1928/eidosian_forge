from enum import Enum
from typing import Dict, List, Sequence, Tuple, cast
import numpy as np
import pandas
from pandas._typing import IndexLabel
from pandas.api.types import is_scalar
class JoinType(Enum):
    """
    An enum that represents the `join_type` argument provided to the algebra operators.

    The enum has 4 values - INNER to represent inner joins, LEFT to represent left joins, RIGHT to
    represent right joins, and OUTER to represent outer joins.
    """
    INNER = 'inner'
    LEFT = 'left'
    RIGHT = 'right'
    OUTER = 'outer'