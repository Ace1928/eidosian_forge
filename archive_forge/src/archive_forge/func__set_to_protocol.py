from traits.api import Any, Bool, HasTraits, Property
from traits.util.api import import_symbol
def _set_to_protocol(self, to_protocol):
    """ Trait property setter. """
    self._to_protocol = to_protocol