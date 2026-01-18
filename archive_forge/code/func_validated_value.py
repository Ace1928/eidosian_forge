import decimal
import math
from copy import copy
from decimal import Decimal
from itertools import product
from typing import TYPE_CHECKING, cast, Dict, Optional, List, Tuple, \
import urllib.parse
from .exceptions import ElementPathError, ElementPathValueError, \
from .helpers import ordinal, get_double, split_function_test
from .etree import is_etree_element, is_etree_document
from .namespaces import XSD_NAMESPACE, XPATH_FUNCTIONS_NAMESPACE, \
from .tree_builders import get_node_tree
from .xpath_nodes import XPathNode, ElementNode, AttributeNode, \
from .datatypes import xsd10_atomic_types, AbstractDateTime, AnyURI, \
from .protocols import ElementProtocol, DocumentProtocol, XsdAttributeProtocol, \
from .sequence_types import is_sequence_type_restriction, match_sequence_type
from .schema_proxy import AbstractSchemaProxy
from .tdop import Token, MultiLabel
from .xpath_context import XPathContext, XPathSchemaContext
def validated_value(self, item: Any, cls: Type[Any], promote: Optional[ClassCheckType]=None, index: Optional[int]=None) -> Any:
    """
        Type promotion checking (see "function conversion rules" in XPath 2.0 language definition)
        """
    if isinstance(item, (cls, ValueToken)):
        return item
    elif promote and isinstance(item, promote):
        return cls(item)
    if self.parser.compatibility_mode:
        if issubclass(cls, str):
            return self.string_value(item)
        elif issubclass(cls, float) or issubclass(float, cls):
            return self.number_value(item)
    if issubclass(cls, XPathToken) or self.parser.version == '1.0':
        code = 'XPTY0004'
    else:
        value = self.data_value(item)
        if isinstance(value, cls):
            return value
        elif isinstance(value, AnyURI) and issubclass(cls, str):
            return cls(value)
        elif isinstance(value, UntypedAtomic):
            try:
                return cls(value)
            except (TypeError, ValueError):
                pass
        code = 'FOTY0012' if value is None else 'XPTY0004'
    if index is None:
        msg = f'item type is {type(item)!r} instead of {cls!r}'
    else:
        msg = f'{ordinal(index + 1)} argument has type {type(item)!r} instead of {cls!r}'
    raise self.error(code, msg)