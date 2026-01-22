import logging
import re
from typing import (
from . import settings
from .utils import choplist
class PSSymbolTable(Generic[_SymbolT]):
    """A utility class for storing PSLiteral/PSKeyword objects.

    Interned objects can be checked its identity with "is" operator.
    """

    def __init__(self, klass: Type[_SymbolT]) -> None:
        self.dict: Dict[PSLiteral.NameType, _SymbolT] = {}
        self.klass: Type[_SymbolT] = klass

    def intern(self, name: PSLiteral.NameType) -> _SymbolT:
        if name in self.dict:
            lit = self.dict[name]
        else:
            lit = self.klass(name)
            self.dict[name] = lit
        return lit