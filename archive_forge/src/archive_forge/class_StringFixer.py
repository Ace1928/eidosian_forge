import os
from typing import SupportsBytes, Type, TypeVar
import nacl.bindings
from nacl import encoding
class StringFixer:

    def __str__(self: SupportsBytes) -> str:
        return str(self.__bytes__())