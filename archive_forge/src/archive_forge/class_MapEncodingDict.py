import json
from decimal import Decimal, ROUND_UP
from types import ModuleType
from typing import cast, Any, Dict, Iterator, Iterable, Optional, Set, Union, Tuple
from xml.etree import ElementTree
from .exceptions import ElementPathError, xpath_error
from .namespaces import XSLT_XQUERY_SERIALIZATION_NAMESPACE
from .datatypes import AnyAtomicType, AnyURI, AbstractDateTime, \
from .xpath_nodes import XPathNode, ElementNode, AttributeNode, DocumentNode, \
from .xpath_tokens import XPathToken, XPathMap, XPathArray
from .protocols import EtreeElementProtocol, LxmlElementProtocol
class MapEncodingDict(dict):

    def __init__(self, items: Any) -> None:
        self[None] = None
        self._items = items

    def items(self) -> Any:
        return self._items