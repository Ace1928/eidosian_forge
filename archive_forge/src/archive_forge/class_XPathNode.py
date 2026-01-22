from importlib import import_module
from urllib.parse import urljoin
from types import ModuleType
from typing import cast, Any, Dict, Iterator, List, MutableMapping, Optional, Tuple, Union
from .datatypes import UntypedAtomic, get_atomic_value, AtomicValueType
from .namespaces import XML_NAMESPACE, XML_BASE, XSI_NIL, \
from .protocols import ElementProtocol, DocumentProtocol, XsdElementProtocol, \
from .helpers import match_wildcard, is_absolute_uri
from .etree import etree_iter_strings, is_etree_element, is_etree_document
class XPathNode:
    """The base class of all XPath nodes. Used only for type checking."""
    kind: str = ''
    children: Optional[List[ChildNodeType]]
    parent: Union['ElementNode', 'DocumentNode', None]
    __slots__ = ('parent', 'position')

    @property
    def attributes(self) -> Optional[List['AttributeNode']]:
        return None

    @property
    def base_uri(self) -> Optional[str]:
        return self.parent.base_uri if self.parent is not None else None

    @property
    def document_uri(self) -> Optional[str]:
        return None

    @property
    def is_id(self) -> Optional[bool]:
        return None

    @property
    def is_idrefs(self) -> Optional[bool]:
        return None

    @property
    def namespace_nodes(self) -> Optional[List['NamespaceNode']]:
        return None

    @property
    def nilled(self) -> Optional[bool]:
        return None

    @property
    def name(self) -> Optional[str]:
        return None

    @property
    def type_name(self) -> Optional[str]:
        return None

    @property
    def string_value(self) -> str:
        raise NotImplementedError()

    @property
    def typed_value(self) -> Optional[AtomicValueType]:
        raise NotImplementedError()
    value: Any
    position: int

    @property
    def root_node(self, namespace: Optional[str]=None) -> 'XPathNode':
        return self if self.parent is None else self.parent.root_node

    def is_schema_node(self) -> Optional[bool]:
        return None

    def match_name(self, name: str, default_namespace: Optional[str]=None) -> bool:
        """
        Returns `True` if the argument is matching the name of the node, `False` otherwise.
        Raises a ValueError if the argument is used, but it's in a wrong format.

        :param name: a fully qualified name, a local name or a wildcard. The accepted         wildcard formats are '*', '*:*', '*:local-name' and '{namespace}*'.
        :param default_namespace: the default namespace for unprefixed names.
        """
        return False