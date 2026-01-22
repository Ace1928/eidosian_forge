import _symtable
from _symtable import (USE, DEF_GLOBAL, DEF_NONLOCAL, DEF_LOCAL, DEF_PARAM,
import weakref
def __idents_matching(self, test_func):
    return tuple((ident for ident in self.get_identifiers() if test_func(self._table.symbols[ident])))