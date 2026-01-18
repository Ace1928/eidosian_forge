import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
def toprettyxml(self, indent='\t', newl='\n', encoding=None, standalone=None):
    if encoding is None:
        writer = io.StringIO()
    else:
        writer = io.TextIOWrapper(io.BytesIO(), encoding=encoding, errors='xmlcharrefreplace', newline='\n')
    if self.nodeType == Node.DOCUMENT_NODE:
        self.writexml(writer, '', indent, newl, encoding, standalone)
    else:
        self.writexml(writer, '', indent, newl)
    if encoding is None:
        return writer.getvalue()
    else:
        return writer.detach().getvalue()