from a string or file.
from xml.dom import xmlbuilder, minidom, Node
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE
from xml.parsers import expat
from xml.dom.minidom import _append_child, _set_attribute_node
from xml.dom.NodeFilter import NodeFilter
def _getDeclarations(self):
    """Re-create the internal subset from the DocumentType node.

        This is only needed if we don't already have the
        internalSubset as a string.
        """
    doctype = self.context.ownerDocument.doctype
    s = ''
    if doctype:
        for i in range(doctype.notations.length):
            notation = doctype.notations.item(i)
            if s:
                s = s + '\n  '
            s = '%s<!NOTATION %s' % (s, notation.nodeName)
            if notation.publicId:
                s = '%s PUBLIC "%s"\n             "%s">' % (s, notation.publicId, notation.systemId)
            else:
                s = '%s SYSTEM "%s">' % (s, notation.systemId)
        for i in range(doctype.entities.length):
            entity = doctype.entities.item(i)
            if s:
                s = s + '\n  '
            s = '%s<!ENTITY %s' % (s, entity.nodeName)
            if entity.publicId:
                s = '%s PUBLIC "%s"\n             "%s"' % (s, entity.publicId, entity.systemId)
            elif entity.systemId:
                s = '%s SYSTEM "%s"' % (s, entity.systemId)
            else:
                s = '%s "%s"' % (s, entity.firstChild.data)
            if entity.notationName:
                s = '%s NOTATION %s' % (s, entity.notationName)
            s = s + '>'
    return s