from typing import cast
from zope.interface import Attribute, Interface, implementer
from twisted.web import sux
def addElement(self, name, defaultUri=None, content=None):
    if isinstance(name, tuple):
        if defaultUri is None:
            defaultUri = name[0]
        child = Element(name, defaultUri)
    else:
        if defaultUri is None:
            defaultUri = self.defaultUri
        child = Element((defaultUri, name), defaultUri)
    self.addChild(child)
    if content:
        child.addContent(content)
    return child