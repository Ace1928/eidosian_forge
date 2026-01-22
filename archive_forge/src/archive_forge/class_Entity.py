import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
class Entity(Identified, Node):
    attributes = None
    nodeType = Node.ENTITY_NODE
    nodeValue = None
    actualEncoding = None
    encoding = None
    version = None

    def __init__(self, name, publicId, systemId, notation):
        self.nodeName = name
        self.notationName = notation
        self.childNodes = NodeList()
        self._identified_mixin_init(publicId, systemId)

    def _get_actualEncoding(self):
        return self.actualEncoding

    def _get_encoding(self):
        return self.encoding

    def _get_version(self):
        return self.version

    def appendChild(self, newChild):
        raise xml.dom.HierarchyRequestErr('cannot append children to an entity node')

    def insertBefore(self, newChild, refChild):
        raise xml.dom.HierarchyRequestErr('cannot insert children below an entity node')

    def removeChild(self, oldChild):
        raise xml.dom.HierarchyRequestErr('cannot remove children from an entity node')

    def replaceChild(self, newChild, oldChild):
        raise xml.dom.HierarchyRequestErr('cannot replace children of an entity node')