from a string or file.
from xml.dom import xmlbuilder, minidom, Node
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE
from xml.parsers import expat
from xml.dom.minidom import _append_child, _set_attribute_node
from xml.dom.NodeFilter import NodeFilter
def _finish_start_element(self, node):
    if self._filter:
        if node is self.document.documentElement:
            return
        filt = self._filter.startContainer(node)
        if filt == FILTER_REJECT:
            Rejecter(self)
        elif filt == FILTER_SKIP:
            Skipper(self)
        else:
            return
        self.curNode = node.parentNode
        node.parentNode.removeChild(node)
        node.unlink()