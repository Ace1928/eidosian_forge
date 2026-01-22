from typing import (
from ._base import BooleanObject, NameObject, NumberObject
from ._data_structures import ArrayObject, DictionaryObject
def _set_int(self, key: str, v: int) -> None:
    self[NameObject(key)] = NumberObject(v)