from typing import (
from ._base import BooleanObject, NameObject, NumberObject
from ._data_structures import ArrayObject, DictionaryObject
def _add_prop_arr(key: str, deft: Optional[ArrayObject]) -> property:
    return property(lambda self: self._get_arr(key, deft), lambda self, v: self._set_arr(key, v), None, f'\n            Returns/Modify the status of {key}, Returns {deft} if not defined\n            ')