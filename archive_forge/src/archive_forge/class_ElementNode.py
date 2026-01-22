from importlib import import_module
from urllib.parse import urljoin
from types import ModuleType
from typing import cast, Any, Dict, Iterator, List, MutableMapping, Optional, Tuple, Union
from .datatypes import UntypedAtomic, get_atomic_value, AtomicValueType
from .namespaces import XML_NAMESPACE, XML_BASE, XSI_NIL, \
from .protocols import ElementProtocol, DocumentProtocol, XsdElementProtocol, \
from .helpers import match_wildcard, is_absolute_uri
from .etree import etree_iter_strings, is_etree_element, is_etree_document
class ElementNode(XPathNode):
    """
    A class for processing XPath element nodes that uses lazy properties to
    diminish the average load for a tree processing.

    :param elem: the wrapped Element or XSD schema/element.
    :param parent: the parent document node or element node.
    :param position: the position of the node in the document.
    :param nsmap: an optional mapping from prefix to namespace URI.
    :param xsd_type: an optional XSD type associated with the element node.
    """
    children: List[ChildNodeType]
    document_uri: None
    kind = 'element'
    elem: Union[ElementProtocol, SchemaElemType]
    nsmap: MutableMapping[Optional[str], str]
    elements: Optional[ElementMapType]
    _namespace_nodes: Optional[List['NamespaceNode']]
    _attributes: Optional[List['AttributeNode']]
    uri: Optional[str] = None
    __slots__ = ('nsmap', 'elem', 'xsd_type', 'elements', '_namespace_nodes', '_attributes', 'children', '__dict__')

    def __init__(self, elem: Union[ElementProtocol, SchemaElemType], parent: Optional[Union['ElementNode', 'DocumentNode']]=None, position: int=1, nsmap: Optional[MutableMapping[Any, str]]=None, xsd_type: Optional[XsdTypeProtocol]=None) -> None:
        self.elem = elem
        self.parent = parent
        self.position = position
        self.xsd_type = xsd_type
        self.elements = None
        self._namespace_nodes = None
        self._attributes = None
        self.children = []
        if nsmap is not None:
            self.nsmap = nsmap
        else:
            try:
                self.nsmap = cast(Dict[Any, str], getattr(elem, 'nsmap'))
            except AttributeError:
                self.nsmap = {}

    def __repr__(self) -> str:
        return '%s(elem=%r)' % (self.__class__.__name__, self.elem)

    def __getitem__(self, i: Union[int, slice]) -> Union[ChildNodeType, List[ChildNodeType]]:
        return self.children[i]

    def __len__(self) -> int:
        return len(self.children)

    def __iter__(self) -> Iterator[ChildNodeType]:
        yield from self.children

    @property
    def value(self) -> Union[ElementProtocol, SchemaElemType]:
        return self.elem

    @property
    def is_id(self) -> bool:
        return False

    @property
    def is_idrefs(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return self.elem.tag

    @property
    def type_name(self) -> Optional[str]:
        if self.xsd_type is None:
            return None
        return self.xsd_type.name

    @property
    def base_uri(self) -> Optional[str]:
        base_uri = self.elem.get(XML_BASE)
        if isinstance(base_uri, str):
            base_uri = base_uri.strip()
        elif base_uri is not None:
            base_uri = ''
        elif self.uri is not None:
            base_uri = self.uri.strip()
        if self.parent is None:
            return base_uri
        elif base_uri is None:
            return self.parent.base_uri
        else:
            return urljoin(self.parent.base_uri or '', base_uri)

    @property
    def nilled(self) -> bool:
        return self.elem.get(XSI_NIL) in ('true', '1')

    @property
    def string_value(self) -> str:
        if self.xsd_type is not None and self.xsd_type.is_element_only():
            return ''.join(etree_iter_strings(self.elem, normalize=True))
        return ''.join(etree_iter_strings(self.elem))

    @property
    def typed_value(self) -> Optional[AtomicValueType]:
        if self.xsd_type is None or self.xsd_type.name in _XSD_SPECIAL_TYPES or self.xsd_type.has_mixed_content():
            return UntypedAtomic(''.join(etree_iter_strings(self.elem)))
        elif self.xsd_type.is_element_only() or self.xsd_type.is_empty():
            return None
        elif self.elem.get(XSI_NIL) and getattr(self.xsd_type.parent, 'nillable', None):
            return None
        if self.elem.text is not None:
            value = self.xsd_type.decode(self.elem.text)
        elif self.elem.get(XSI_NIL) in ('1', 'true'):
            return ''
        else:
            value = self.xsd_type.decode('')
        return cast(Optional[AtomicValueType], value)

    @property
    def namespace_nodes(self) -> List['NamespaceNode']:
        if self._namespace_nodes is None:
            position = self.position + 1
            self._namespace_nodes = [NamespaceNode('xml', XML_NAMESPACE, self, position)]
            position += 1
            if self.nsmap:
                for pfx, uri in self.nsmap.items():
                    if pfx != 'xml':
                        self._namespace_nodes.append(NamespaceNode(pfx, uri, self, position))
                        position += 1
        return self._namespace_nodes

    @property
    def attributes(self) -> List['AttributeNode']:
        if self._attributes is None:
            position = self.position + len(self.nsmap) + int('xml' not in self.nsmap)
            self._attributes = [AttributeNode(name, cast(str, value), self, pos) for pos, (name, value) in enumerate(self.elem.attrib.items(), position)]
        return self._attributes

    @property
    def path(self) -> str:
        """Returns an absolute path for the node."""
        path = []
        item: Any = self
        while True:
            if isinstance(item, ElementNode):
                path.append(item.elem.tag)
            item = item.parent
            if item is None:
                return '/{}'.format('/'.join(reversed(path)))

    def is_schema_node(self) -> bool:
        return hasattr(self.elem, 'name') and hasattr(self.elem, 'type')
    is_schema_element = is_schema_node

    def match_name(self, name: str, default_namespace: Optional[str]=None) -> bool:
        if '*' in name:
            return match_wildcard(self.elem.tag, name)
        elif not name:
            return not self.elem.tag
        elif hasattr(self.elem, 'type'):
            return cast(XsdElementProtocol, self.elem).is_matching(name, default_namespace)
        elif name[0] == '{' or default_namespace is None:
            return self.elem.tag == name
        if None in self.nsmap:
            default_namespace = self.nsmap[None]
        if default_namespace:
            return self.elem.tag == '{%s}%s' % (default_namespace, name)
        return self.elem.tag == name

    def get_element_node(self, elem: Union[ElementProtocol, SchemaElemType]) -> Optional['ElementNode']:
        if self.elements is not None:
            return self.elements.get(elem)
        for node in self.iter():
            if isinstance(node, ElementNode) and elem is node.elem:
                return node
        else:
            return None

    def iter(self) -> Iterator[XPathNode]:
        yield self
        iterators: List[Any] = []
        children: Iterator[Any] = iter(self.children)
        if self._namespace_nodes:
            yield from self._namespace_nodes
        if self._attributes:
            yield from self._attributes
        while True:
            for child in children:
                yield child
                if isinstance(child, ElementNode):
                    if child._namespace_nodes:
                        yield from child._namespace_nodes
                    if child._attributes:
                        yield from child._attributes
                    if child.children:
                        iterators.append(children)
                        children = iter(child.children)
                        break
            else:
                try:
                    children = iterators.pop()
                except IndexError:
                    return

    def iter_document(self) -> Iterator[XPathNode]:
        yield self
        yield from self.namespace_nodes
        yield from self.attributes
        for child in self:
            if isinstance(child, ElementNode):
                yield from child.iter()
            else:
                yield child

    def iter_descendants(self, with_self: bool=True) -> Iterator[ChildNodeType]:
        if with_self:
            yield self
        iterators: List[Any] = []
        children: Iterator[Any] = iter(self.children)
        while True:
            for child in children:
                yield child
                if isinstance(child, ElementNode) and child.children:
                    iterators.append(children)
                    children = iter(child.children)
                    break
            else:
                try:
                    children = iterators.pop()
                except IndexError:
                    return