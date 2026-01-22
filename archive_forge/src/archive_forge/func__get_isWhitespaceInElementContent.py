import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
def _get_isWhitespaceInElementContent(self):
    if self.data.strip():
        return False
    elem = _get_containing_element(self)
    if elem is None:
        return False
    info = self.ownerDocument._get_elem_info(elem)
    if info is None:
        return False
    else:
        return info.isElementContent()