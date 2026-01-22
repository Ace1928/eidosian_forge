from importlib import import_module
from urllib.parse import urljoin
from types import ModuleType
from typing import cast, Any, Dict, Iterator, List, MutableMapping, Optional, Tuple, Union
from .datatypes import UntypedAtomic, get_atomic_value, AtomicValueType
from .namespaces import XML_NAMESPACE, XML_BASE, XSI_NIL, \
from .protocols import ElementProtocol, DocumentProtocol, XsdElementProtocol, \
from .helpers import match_wildcard, is_absolute_uri
from .etree import etree_iter_strings, is_etree_element, is_etree_document
class ProcessingInstructionNode(XPathNode):
    """
    A class for XPath processing instructions nodes.

    :param elem: the wrapped Processing Instruction Element.
    :param parent: the parent element node.
    :param position: the position of the node in the document.
    """
    attributes: None
    children: None = None
    document_uri: None
    is_id: None
    is_idrefs: None
    namespace_nodes: None
    nilled: None
    type_name: None
    kind = 'processing-instruction'
    __slots__ = ('elem',)

    def __init__(self, elem: ElementProtocol, parent: Union['ElementNode', 'DocumentNode', None]=None, position: int=1) -> None:
        self.elem = elem
        self.parent = parent
        self.position = position

    def __repr__(self) -> str:
        return '%s(elem=%r)' % (self.__class__.__name__, self.elem)

    @property
    def value(self) -> ElementProtocol:
        return self.elem

    @property
    def name(self) -> str:
        try:
            return cast(str, self.elem.target)
        except AttributeError:
            return cast(str, self.elem.text).split(' ', maxsplit=1)[0]

    @property
    def string_value(self) -> str:
        if hasattr(self.elem, 'target'):
            return self.elem.text or ''
        try:
            return cast(str, self.elem.text).split(' ', maxsplit=1)[1]
        except IndexError:
            return ''

    @property
    def typed_value(self) -> str:
        return self.string_value