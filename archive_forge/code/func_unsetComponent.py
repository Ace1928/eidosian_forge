from io import StringIO
from typing import Dict
from zope.interface import declarations, interface
from zope.interface.adapter import AdapterRegistry
from twisted.python import reflect
def unsetComponent(self, interfaceClass):
    """Remove my component specified by the given interface class."""
    del self._adapterCache[reflect.qual(interfaceClass)]