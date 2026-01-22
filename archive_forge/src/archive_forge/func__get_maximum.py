from . import _gi
from ._constants import \
def _get_maximum(self):
    return self._max_value_lookup.get(self.type, None)