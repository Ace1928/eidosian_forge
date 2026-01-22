from __future__ import absolute_import, division, unicode_literals
from xml.dom import minidom, Node
import weakref
from . import base
from .. import constants
from ..constants import namespaces
from .._utils import moduleFactoryFactory
class NodeBuilder(base.Node):

    def __init__(self, element):
        base.Node.__init__(self, element.nodeName)
        self.element = element
    namespace = property(lambda self: hasattr(self.element, 'namespaceURI') and self.element.namespaceURI or None)

    def appendChild(self, node):
        node.parent = self
        self.element.appendChild(node.element)

    def insertText(self, data, insertBefore=None):
        text = self.element.ownerDocument.createTextNode(data)
        if insertBefore:
            self.element.insertBefore(text, insertBefore.element)
        else:
            self.element.appendChild(text)

    def insertBefore(self, node, refNode):
        self.element.insertBefore(node.element, refNode.element)
        node.parent = self

    def removeChild(self, node):
        if node.element.parentNode == self.element:
            self.element.removeChild(node.element)
        node.parent = None

    def reparentChildren(self, newParent):
        while self.element.hasChildNodes():
            child = self.element.firstChild
            self.element.removeChild(child)
            newParent.element.appendChild(child)
        self.childNodes = []

    def getAttributes(self):
        return AttrList(self.element)

    def setAttributes(self, attributes):
        if attributes:
            for name, value in list(attributes.items()):
                if isinstance(name, tuple):
                    if name[0] is not None:
                        qualifiedName = name[0] + ':' + name[1]
                    else:
                        qualifiedName = name[1]
                    self.element.setAttributeNS(name[2], qualifiedName, value)
                else:
                    self.element.setAttribute(name, value)
    attributes = property(getAttributes, setAttributes)

    def cloneNode(self):
        return NodeBuilder(self.element.cloneNode(False))

    def hasContent(self):
        return self.element.hasChildNodes()

    def getNameTuple(self):
        if self.namespace is None:
            return (namespaces['html'], self.name)
        else:
            return (self.namespace, self.name)
    nameTuple = property(getNameTuple)