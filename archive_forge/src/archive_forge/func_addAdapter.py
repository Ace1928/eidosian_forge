from io import StringIO
from typing import Dict
from zope.interface import declarations, interface
from zope.interface.adapter import AdapterRegistry
from twisted.python import reflect
def addAdapter(self, adapterClass, ignoreClass=0):
    """Utility method that calls addComponent.  I take an adapter class and
        instantiate it with myself as the first argument.

        @return: The adapter instantiated.
        """
    adapt = adapterClass(self)
    self.addComponent(adapt, ignoreClass)
    return adapt