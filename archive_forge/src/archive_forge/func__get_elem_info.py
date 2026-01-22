import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
def _get_elem_info(self, element):
    if element.namespaceURI:
        key = (element.namespaceURI, element.localName)
    else:
        key = element.tagName
    return self._elem_info.get(key)