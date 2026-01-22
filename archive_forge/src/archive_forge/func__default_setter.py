from . import _gi
from ._constants import \
def _default_setter(self, instance, value):
    setattr(instance, '_property_helper_' + self.name, value)