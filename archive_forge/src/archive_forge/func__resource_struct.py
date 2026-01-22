from . import schema
from .jsonutil import get_column
from .search import Search
def _resource_struct(self, name):
    return self._intf._struct