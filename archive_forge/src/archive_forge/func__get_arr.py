from typing import (
from ._base import BooleanObject, NameObject, NumberObject
from ._data_structures import ArrayObject, DictionaryObject
def _get_arr(self, key: str, deft: Optional[List[Any]]) -> NumberObject:
    return self.get(key, None if deft is None else ArrayObject(deft))