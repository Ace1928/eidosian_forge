from __future__ import annotations
import operator
import typing
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import Tuple
from typing import Type
def tuplegetter(*indexes: int) -> _TupleGetterType:
    if len(indexes) != 1:
        for i in range(1, len(indexes)):
            if indexes[i - 1] != indexes[i] - 1:
                return operator.itemgetter(*indexes)
    return operator.itemgetter(slice(indexes[0], indexes[-1] + 1))