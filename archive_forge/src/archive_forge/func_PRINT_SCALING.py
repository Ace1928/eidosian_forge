from typing import (
from ._base import BooleanObject, NameObject, NumberObject
from ._data_structures import ArrayObject, DictionaryObject
@property
def PRINT_SCALING(self) -> NameObject:
    return NameObject('/PrintScaling')