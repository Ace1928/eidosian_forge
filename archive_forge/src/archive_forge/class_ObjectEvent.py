from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
@implementer(IObjectEvent)
class ObjectEvent:

    def __init__(self, object):
        self.object = object