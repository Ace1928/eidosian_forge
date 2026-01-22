from . import _gi
from ._constants import \
def _readonly_setter(self, instance, value):
    self._exc = TypeError('%s property of %s is read-only' % (self.name, type(instance).__name__))