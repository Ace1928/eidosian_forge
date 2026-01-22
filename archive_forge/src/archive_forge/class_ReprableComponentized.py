from io import StringIO
from typing import Dict
from zope.interface import declarations, interface
from zope.interface.adapter import AdapterRegistry
from twisted.python import reflect
class ReprableComponentized(Componentized):

    def __init__(self):
        Componentized.__init__(self)

    def __repr__(self) -> str:
        from pprint import pprint
        sio = StringIO()
        pprint(self._adapterCache, sio)
        return sio.getvalue()