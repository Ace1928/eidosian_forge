from a string or file.
from xml.dom import xmlbuilder, minidom, Node
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE
from xml.parsers import expat
from xml.dom.minidom import _append_child, _set_attribute_node
from xml.dom.NodeFilter import NodeFilter
class FilterVisibilityController(object):
    """Wrapper around a DOMBuilderFilter which implements the checks
    to make the whatToShow filter attribute work."""
    __slots__ = ('filter',)

    def __init__(self, filter):
        self.filter = filter

    def startContainer(self, node):
        mask = self._nodetype_mask[node.nodeType]
        if self.filter.whatToShow & mask:
            val = self.filter.startContainer(node)
            if val == FILTER_INTERRUPT:
                raise ParseEscape
            if val not in _ALLOWED_FILTER_RETURNS:
                raise ValueError('startContainer() returned illegal value: ' + repr(val))
            return val
        else:
            return FILTER_ACCEPT

    def acceptNode(self, node):
        mask = self._nodetype_mask[node.nodeType]
        if self.filter.whatToShow & mask:
            val = self.filter.acceptNode(node)
            if val == FILTER_INTERRUPT:
                raise ParseEscape
            if val == FILTER_SKIP:
                parent = node.parentNode
                for child in node.childNodes[:]:
                    parent.appendChild(child)
                return FILTER_REJECT
            if val not in _ALLOWED_FILTER_RETURNS:
                raise ValueError('acceptNode() returned illegal value: ' + repr(val))
            return val
        else:
            return FILTER_ACCEPT
    _nodetype_mask = {Node.ELEMENT_NODE: NodeFilter.SHOW_ELEMENT, Node.ATTRIBUTE_NODE: NodeFilter.SHOW_ATTRIBUTE, Node.TEXT_NODE: NodeFilter.SHOW_TEXT, Node.CDATA_SECTION_NODE: NodeFilter.SHOW_CDATA_SECTION, Node.ENTITY_REFERENCE_NODE: NodeFilter.SHOW_ENTITY_REFERENCE, Node.ENTITY_NODE: NodeFilter.SHOW_ENTITY, Node.PROCESSING_INSTRUCTION_NODE: NodeFilter.SHOW_PROCESSING_INSTRUCTION, Node.COMMENT_NODE: NodeFilter.SHOW_COMMENT, Node.DOCUMENT_NODE: NodeFilter.SHOW_DOCUMENT, Node.DOCUMENT_TYPE_NODE: NodeFilter.SHOW_DOCUMENT_TYPE, Node.DOCUMENT_FRAGMENT_NODE: NodeFilter.SHOW_DOCUMENT_FRAGMENT, Node.NOTATION_NODE: NodeFilter.SHOW_NOTATION}