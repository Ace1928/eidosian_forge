import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
class CDATASection(Text):
    __slots__ = ()
    nodeType = Node.CDATA_SECTION_NODE
    nodeName = '#cdata-section'

    def writexml(self, writer, indent='', addindent='', newl=''):
        if self.data.find(']]>') >= 0:
            raise ValueError("']]>' not allowed in a CDATA section")
        writer.write('<![CDATA[%s]]>' % self.data)