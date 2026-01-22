import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
class ProcessingInstruction(Childless, Node):
    nodeType = Node.PROCESSING_INSTRUCTION_NODE
    __slots__ = ('target', 'data')

    def __init__(self, target, data):
        self.target = target
        self.data = data

    def _get_nodeValue(self):
        return self.data

    def _set_nodeValue(self, value):
        self.data = value
    nodeValue = property(_get_nodeValue, _set_nodeValue)

    def _get_nodeName(self):
        return self.target

    def _set_nodeName(self, value):
        self.target = value
    nodeName = property(_get_nodeName, _set_nodeName)

    def writexml(self, writer, indent='', addindent='', newl=''):
        writer.write('%s<?%s %s?>%s' % (indent, self.target, self.data, newl))