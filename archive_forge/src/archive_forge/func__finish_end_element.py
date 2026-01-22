from a string or file.
from xml.dom import xmlbuilder, minidom, Node
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE
from xml.parsers import expat
from xml.dom.minidom import _append_child, _set_attribute_node
from xml.dom.NodeFilter import NodeFilter
def _finish_end_element(self, curNode):
    info = self._elem_info.get(curNode.tagName)
    if info:
        self._handle_white_text_nodes(curNode, info)
    if self._filter:
        if curNode is self.document.documentElement:
            return
        if self._filter.acceptNode(curNode) == FILTER_REJECT:
            self.curNode.removeChild(curNode)
            curNode.unlink()