import sys
import threading
from .__wrapt__ import ObjectProxy
def _self_create_module(self, spec):
    return self.__wrapped__.create_module(spec)