import string
from xml.dom import Node
class _implementation:
    """Implementation class for C14N. This accompanies a node during it's
    processing and includes the parameters and processing state."""
    handlers = {}

    def __init__(self, node, write, **kw):
        """Create and run the implementation."""
        self.write = write
        self.subset = kw.get('subset')
        self.comments = kw.get('comments', 0)
        self.unsuppressedPrefixes = kw.get('unsuppressedPrefixes')
        nsdict = kw.get('nsdict', {'xml': XMLNS.XML, 'xmlns': XMLNS.BASE})
        self.state = (nsdict, {'xml': ''}, {}, {})
        if node.nodeType == Node.DOCUMENT_NODE:
            self._do_document(node)
        elif node.nodeType == Node.ELEMENT_NODE:
            self.documentOrder = _Element
            if not _inclusive(self):
                inherited, unused = _inclusiveNamespacePrefixes(node, self._inherit_context(node), self.unsuppressedPrefixes)
                self._do_element(node, inherited, unused=unused)
            else:
                inherited = self._inherit_context(node)
                self._do_element(node, inherited)
        elif node.nodeType == Node.DOCUMENT_TYPE_NODE:
            pass
        else:
            raise TypeError(str(node))

    def _inherit_context(self, node):
        """_inherit_context(self, node) -> list
        Scan ancestors of attribute and namespace context.  Used only
        for single element node canonicalization, not for subset
        canonicalization."""
        xmlattrs = filter(_IN_XML_NS, _attrs(node))
        inherited, parent = ([], node.parentNode)
        while parent and parent.nodeType == Node.ELEMENT_NODE:
            for a in filter(_IN_XML_NS, _attrs(parent)):
                n = a.localName
                if n not in xmlattrs:
                    xmlattrs.append(n)
                    inherited.append(a)
            parent = parent.parentNode
        return inherited

    def _do_document(self, node):
        """_do_document(self, node) -> None
        Process a document node. documentOrder holds whether the document
        element has been encountered such that PIs/comments can be written
        as specified."""
        self.documentOrder = _LesserElement
        for child in node.childNodes:
            if child.nodeType == Node.ELEMENT_NODE:
                self.documentOrder = _Element
                self._do_element(child)
                self.documentOrder = _GreaterElement
            elif child.nodeType == Node.PROCESSING_INSTRUCTION_NODE:
                self._do_pi(child)
            elif child.nodeType == Node.COMMENT_NODE:
                self._do_comment(child)
            elif child.nodeType == Node.DOCUMENT_TYPE_NODE:
                pass
            else:
                raise TypeError(str(child))
    handlers[Node.DOCUMENT_NODE] = _do_document

    def _do_text(self, node):
        """_do_text(self, node) -> None
        Process a text or CDATA node.  Render various special characters
        as their C14N entity representations."""
        if not _in_subset(self.subset, node):
            return
        s = string.replace(node.data, '&', '&amp;')
        s = string.replace(s, '<', '&lt;')
        s = string.replace(s, '>', '&gt;')
        s = string.replace(s, '\r', '&#xD;')
        if s:
            self.write(s)
    handlers[Node.TEXT_NODE] = _do_text
    handlers[Node.CDATA_SECTION_NODE] = _do_text

    def _do_pi(self, node):
        """_do_pi(self, node) -> None
        Process a PI node. Render a leading or trailing #xA if the
        document order of the PI is greater or lesser (respectively)
        than the document element.
        """
        if not _in_subset(self.subset, node):
            return
        W = self.write
        if self.documentOrder == _GreaterElement:
            W('\n')
        W('<?')
        W(node.nodeName)
        s = node.data
        if s:
            W(' ')
            W(s)
        W('?>')
        if self.documentOrder == _LesserElement:
            W('\n')
    handlers[Node.PROCESSING_INSTRUCTION_NODE] = _do_pi

    def _do_comment(self, node):
        """_do_comment(self, node) -> None
        Process a comment node. Render a leading or trailing #xA if the
        document order of the comment is greater or lesser (respectively)
        than the document element.
        """
        if not _in_subset(self.subset, node):
            return
        if self.comments:
            W = self.write
            if self.documentOrder == _GreaterElement:
                W('\n')
            W('<!--')
            W(node.data)
            W('-->')
            if self.documentOrder == _LesserElement:
                W('\n')
    handlers[Node.COMMENT_NODE] = _do_comment

    def _do_attr(self, n, value):
        """'_do_attr(self, node) -> None
        Process an attribute."""
        W = self.write
        W(' ')
        W(n)
        W('="')
        s = string.replace(value, '&', '&amp;')
        s = string.replace(s, '<', '&lt;')
        s = string.replace(s, '"', '&quot;')
        s = string.replace(s, '\t', '&#x9')
        s = string.replace(s, '\n', '&#xA')
        s = string.replace(s, '\r', '&#xD')
        W(s)
        W('"')

    def _do_element(self, node, initial_other_attrs=[], unused=None):
        """_do_element(self, node, initial_other_attrs = [], unused = {}) -> None
        Process an element (and its children)."""
        ns_parent, ns_rendered, xml_attrs = (self.state[0], self.state[1].copy(), self.state[2].copy())
        ns_unused_inherited = unused
        if unused is None:
            ns_unused_inherited = self.state[3].copy()
        ns_local = ns_parent.copy()
        inclusive = _inclusive(self)
        xml_attrs_local = {}
        other_attrs = []
        in_subset = _in_subset(self.subset, node)
        for a in initial_other_attrs + _attrs(node):
            if a.namespaceURI == XMLNS.BASE:
                n = a.nodeName
                if n == 'xmlns:':
                    n = 'xmlns'
                ns_local[n] = a.nodeValue
            elif a.namespaceURI == XMLNS.XML:
                if inclusive or (in_subset and _in_subset(self.subset, a)):
                    xml_attrs_local[a.nodeName] = a
            elif _in_subset(self.subset, a):
                other_attrs.append(a)
            xml_attrs.update(xml_attrs_local)
        W, name = (self.write, None)
        if in_subset:
            name = node.nodeName
            if not inclusive:
                if node.prefix is not None:
                    prefix = 'xmlns:%s' % node.prefix
                else:
                    prefix = 'xmlns'
                if not ns_rendered.has_key(prefix) and (not ns_local.has_key(prefix)):
                    if not ns_unused_inherited.has_key(prefix):
                        raise RuntimeError('For exclusive c14n, unable to map prefix "%s" in %s' % (prefix, node))
                    ns_local[prefix] = ns_unused_inherited[prefix]
                    del ns_unused_inherited[prefix]
            W('<')
            W(name)
            ns_to_render = []
            for n, v in ns_local.items():
                if n == 'xmlns' and v in [XMLNS.BASE, ''] and (ns_rendered.get('xmlns') in [XMLNS.BASE, '', None]):
                    continue
                if n in ['xmlns:xml', 'xml'] and v in ['http://www.w3.org/XML/1998/namespace']:
                    continue
                if (n, v) not in ns_rendered.items():
                    if inclusive or _utilized(n, node, other_attrs, self.unsuppressedPrefixes):
                        ns_to_render.append((n, v))
                    elif not inclusive:
                        ns_unused_inherited[n] = v
            ns_to_render.sort(_sorter_ns)
            for n, v in ns_to_render:
                self._do_attr(n, v)
                ns_rendered[n] = v
            if not inclusive or _in_subset(self.subset, node.parentNode):
                other_attrs.extend(xml_attrs_local.values())
            else:
                other_attrs.extend(xml_attrs.values())
            other_attrs.sort(_sorter)
            for a in other_attrs:
                self._do_attr(a.nodeName, a.value)
            W('>')
        state, self.state = (self.state, (ns_local, ns_rendered, xml_attrs, ns_unused_inherited))
        for c in _children(node):
            _implementation.handlers[c.nodeType](self, c)
        self.state = state
        if name:
            W('</%s>' % name)
    handlers[Node.ELEMENT_NODE] = _do_element