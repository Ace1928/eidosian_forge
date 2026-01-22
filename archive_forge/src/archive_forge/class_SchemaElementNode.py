from importlib import import_module
from urllib.parse import urljoin
from types import ModuleType
from typing import cast, Any, Dict, Iterator, List, MutableMapping, Optional, Tuple, Union
from .datatypes import UntypedAtomic, get_atomic_value, AtomicValueType
from .namespaces import XML_NAMESPACE, XML_BASE, XSI_NIL, \
from .protocols import ElementProtocol, DocumentProtocol, XsdElementProtocol, \
from .helpers import match_wildcard, is_absolute_uri
from .etree import etree_iter_strings, is_etree_element, is_etree_document
class SchemaElementNode(ElementNode):
    """
    An element node class for wrapping the XSD schema and its elements.
    The resulting structure can be a tree or a set of disjoint trees.
    With more roots only one of them is the schema node.
    """
    __slots__ = ()
    ref: Optional['SchemaElementNode'] = None
    elem: SchemaElemType

    def __iter__(self) -> Iterator[ChildNodeType]:
        if self.ref is None:
            yield from self.children
        else:
            yield from self.ref.children

    @property
    def attributes(self) -> List['AttributeNode']:
        if self._attributes is None:
            position = self.position + len(self.nsmap) + int('xml' not in self.nsmap)
            self._attributes = [AttributeNode(name, attr, self, pos, attr.type) for pos, (name, attr) in enumerate(self.elem.attrib.items(), position)]
        return self._attributes

    @property
    def base_uri(self) -> Optional[str]:
        base_uri = self.uri.strip() if self.uri is not None else None
        if self.parent is None:
            return base_uri
        elif base_uri is None:
            return self.parent.base_uri
        else:
            return urljoin(self.parent.base_uri or '', base_uri)

    @property
    def string_value(self) -> str:
        if not hasattr(self.elem, 'type'):
            return ''
        schema_node = cast(XsdElementProtocol, self.elem)
        return str(get_atomic_value(schema_node.type))

    @property
    def typed_value(self) -> Optional[AtomicValueType]:
        if not hasattr(self.elem, 'type'):
            return UntypedAtomic('')
        schema_node = cast(XsdElementProtocol, self.elem)
        return get_atomic_value(schema_node.type)

    def iter(self) -> Iterator[XPathNode]:
        yield self
        iterators: List[Any] = []
        children: Iterator[Any] = iter(self.children)
        if self._namespace_nodes:
            yield from self._namespace_nodes
        if self._attributes:
            yield from self._attributes
        elements = {self}
        while True:
            for child in children:
                if child in elements:
                    continue
                yield child
                elements.add(child)
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

    def iter_descendants(self, with_self: bool=True) -> Iterator[ChildNodeType]:
        if with_self:
            yield self
        iterators: List[Any] = []
        children: Iterator[Any] = iter(self.children)
        elements = {self}
        while True:
            for child in children:
                if child.ref is not None:
                    child = child.ref
                if child in elements:
                    continue
                yield child
                elements.add(child)
                if child.children:
                    iterators.append(children)
                    children = iter(child.children)
                    break
            else:
                try:
                    children = iterators.pop()
                except IndexError:
                    return