from xml.sax.handler import ContentHandler
from lxml import etree
from lxml.etree import ElementTree, SubElement
from lxml.etree import Comment, ProcessingInstruction
class ElementTreeProducer:
    """Produces SAX events for an element and children.
    """

    def __init__(self, element_or_tree, content_handler):
        try:
            element = element_or_tree.getroot()
        except AttributeError:
            element = element_or_tree
        self._element = element
        self._content_handler = content_handler
        from xml.sax.xmlreader import AttributesNSImpl as attr_class
        self._attr_class = attr_class
        self._empty_attributes = attr_class({}, {})

    def saxify(self):
        self._content_handler.startDocument()
        element = self._element
        if hasattr(element, 'getprevious'):
            siblings = []
            sibling = element.getprevious()
            while getattr(sibling, 'tag', None) is ProcessingInstruction:
                siblings.append(sibling)
                sibling = sibling.getprevious()
            for sibling in siblings[::-1]:
                self._recursive_saxify(sibling, {})
        self._recursive_saxify(element, {})
        if hasattr(element, 'getnext'):
            sibling = element.getnext()
            while getattr(sibling, 'tag', None) is ProcessingInstruction:
                self._recursive_saxify(sibling, {})
                sibling = sibling.getnext()
        self._content_handler.endDocument()

    def _recursive_saxify(self, element, parent_nsmap):
        content_handler = self._content_handler
        tag = element.tag
        if tag is Comment or tag is ProcessingInstruction:
            if tag is ProcessingInstruction:
                content_handler.processingInstruction(element.target, element.text)
            tail = element.tail
            if tail:
                content_handler.characters(tail)
            return
        element_nsmap = element.nsmap
        new_prefixes = []
        if element_nsmap != parent_nsmap:
            for prefix, ns_uri in element_nsmap.items():
                if parent_nsmap.get(prefix) != ns_uri:
                    new_prefixes.append((prefix, ns_uri))
        attribs = element.items()
        if attribs:
            attr_values = {}
            attr_qnames = {}
            for attr_ns_name, value in attribs:
                attr_ns_tuple = _getNsTag(attr_ns_name)
                attr_values[attr_ns_tuple] = value
                attr_qnames[attr_ns_tuple] = self._build_qname(attr_ns_tuple[0], attr_ns_tuple[1], element_nsmap, preferred_prefix=None, is_attribute=True)
            sax_attributes = self._attr_class(attr_values, attr_qnames)
        else:
            sax_attributes = self._empty_attributes
        ns_uri, local_name = _getNsTag(tag)
        qname = self._build_qname(ns_uri, local_name, element_nsmap, element.prefix, is_attribute=False)
        for prefix, uri in new_prefixes:
            content_handler.startPrefixMapping(prefix, uri)
        content_handler.startElementNS((ns_uri, local_name), qname, sax_attributes)
        text = element.text
        if text:
            content_handler.characters(text)
        for child in element:
            self._recursive_saxify(child, element_nsmap)
        content_handler.endElementNS((ns_uri, local_name), qname)
        for prefix, uri in new_prefixes:
            content_handler.endPrefixMapping(prefix)
        tail = element.tail
        if tail:
            content_handler.characters(tail)

    def _build_qname(self, ns_uri, local_name, nsmap, preferred_prefix, is_attribute):
        if ns_uri is None:
            return local_name
        if not is_attribute and nsmap.get(preferred_prefix) == ns_uri:
            prefix = preferred_prefix
        else:
            candidates = [pfx for pfx, uri in nsmap.items() if pfx is not None and uri == ns_uri]
            prefix = candidates[0] if len(candidates) == 1 else min(candidates) if candidates else None
        if prefix is None:
            return local_name
        return prefix + ':' + local_name