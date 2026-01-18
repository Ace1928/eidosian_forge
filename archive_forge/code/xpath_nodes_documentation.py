from importlib import import_module
from urllib.parse import urljoin
from types import ModuleType
from typing import cast, Any, Dict, Iterator, List, MutableMapping, Optional, Tuple, Union
from .datatypes import UntypedAtomic, get_atomic_value, AtomicValueType
from .namespaces import XML_NAMESPACE, XML_BASE, XSI_NIL, \
from .protocols import ElementProtocol, DocumentProtocol, XsdElementProtocol, \
from .helpers import match_wildcard, is_absolute_uri
from .etree import etree_iter_strings, is_etree_element, is_etree_document

        Build a `DocumentNode` from a tree based on an ElementNode.

        :param root_node: the root element node.
        :param replace: if `True` the root element is replaced by a document node.         This is usually useful for extended data models (more element children, text nodes).
        