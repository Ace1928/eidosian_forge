from typing import ClassVar, Dict, Tuple
from ..namespaces import XPATH_MAP_FUNCTIONS_NAMESPACE, \
from ..datatypes import QName
from ..xpath30 import XPath30Parser
class XPath31Parser(XPath30Parser):
    """
    XPath 3.1 expression parser class.
    """
    version = '3.1'
    DEFAULT_NAMESPACES: ClassVar[Dict[str, str]] = {'map': XPATH_MAP_FUNCTIONS_NAMESPACE, 'array': XPATH_ARRAY_FUNCTIONS_NAMESPACE, **XPath30Parser.DEFAULT_NAMESPACES}
    RESERVED_FUNCTION_NAMES = {'array', 'attribute', 'comment', 'document-node', 'element', 'empty-sequence', 'function', 'if', 'item', 'map', 'namespace-node', 'node', 'processing-instruction', 'schema-attribute', 'schema-element', 'switch', 'text', 'typeswitch'}
    function_signatures: Dict[Tuple[QName, int], str] = XPath30Parser.function_signatures.copy()