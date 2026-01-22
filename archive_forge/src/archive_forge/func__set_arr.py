from typing import (
from ._base import BooleanObject, NameObject, NumberObject
from ._data_structures import ArrayObject, DictionaryObject
def _set_arr(self, key: str, v: Optional[ArrayObject]) -> None:
    if v is None:
        try:
            del self[NameObject(key)]
        except KeyError:
            pass
        return
    if not isinstance(v, ArrayObject):
        raise ValueError('ArrayObject is expected')
    self[NameObject(key)] = v