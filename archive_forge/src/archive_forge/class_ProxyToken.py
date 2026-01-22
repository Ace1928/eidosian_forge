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
class ProxyToken(XPathToken):
    """
    A proxy token for resolving or calling namespace related functions.

    It cannot handle symbols associated with tokens that are not related
    to namespaces, like operators, type tests or axes. Those tokens can
    have also different binding powers, so handling disambiguation could
    be impracticable.
    """
    label = 'proxy function'

    def nud(self) -> XPathToken:
        lookup_name = f'{{{self.namespace or XPATH_FUNCTIONS_NAMESPACE}}}{self.value}'
        try:
            token = self.parser.symbol_table[lookup_name](self.parser)
        except KeyError:
            if self.namespace == XSD_NAMESPACE:
                msg = f'unknown constructor function {self.symbol!r}'
            else:
                msg = f'unknown function {self.symbol!r}'
            raise self.error('XPST0017', msg) from None
        else:
            if self.parser.next_token.symbol == '#':
                if self.parser.version >= '2.0':
                    return token
            return token.nud()