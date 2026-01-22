import sys
import threading
from .__wrapt__ import ObjectProxy
def _self_set_loader(self, module):

    class UNDEFINED:
        pass
    if getattr(module, '__loader__', UNDEFINED) in (None, self):
        try:
            module.__loader__ = self.__wrapped__
        except AttributeError:
            pass
    if getattr(module, '__spec__', None) is not None and getattr(module.__spec__, 'loader', None) is self:
        module.__spec__.loader = self.__wrapped__