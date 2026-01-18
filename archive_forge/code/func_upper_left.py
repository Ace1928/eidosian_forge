from typing import Any, Tuple, Union
from ._base import FloatObject, NumberObject
from ._data_structures import ArrayObject
@upper_left.setter
def upper_left(self, value: Tuple[float, float]) -> None:
    self[0], self[3] = (self._ensure_is_number(x) for x in value)