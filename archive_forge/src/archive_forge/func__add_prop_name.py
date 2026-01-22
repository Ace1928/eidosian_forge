from typing import (
from ._base import BooleanObject, NameObject, NumberObject
from ._data_structures import ArrayObject, DictionaryObject
def _add_prop_name(key: str, lst: List[str], deft: Optional[NameObject]) -> property:
    return property(lambda self: self._get_name(key, deft), lambda self, v: self._set_name(key, lst, v), None, f'\n            Returns/Modify the status of {key}, Returns {deft} if not defined.\n            Acceptable values: {lst}\n            ')