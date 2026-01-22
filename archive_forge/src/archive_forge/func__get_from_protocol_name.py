from traits.api import Any, Bool, HasTraits, Property
from traits.util.api import import_symbol
def _get_from_protocol_name(self):
    return self._get_type_name(self._from_protocol)