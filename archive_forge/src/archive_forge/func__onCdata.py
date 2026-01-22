from typing import cast
from zope.interface import Attribute, Interface, implementer
from twisted.web import sux
def _onCdata(self, data):
    if self.currElem != None:
        self.currElem.addContent(data)