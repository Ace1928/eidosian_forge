import json
import locale
import math
import pathlib
import random
import re
from datetime import datetime, timedelta
from decimal import Decimal
from itertools import product
from urllib.request import urlopen
from urllib.parse import urlsplit
from ..datatypes import AnyAtomicType, AbstractBinary, AbstractDateTime, \
from ..exceptions import ElementPathTypeError
from ..helpers import WHITESPACES_PATTERN, is_xml_codepoint, \
from ..namespaces import XPATH_FUNCTIONS_NAMESPACE, XML_BASE
from ..etree import etree_iter_strings, is_etree_element
from ..collations import CollationManager
from ..compare import get_key_function, same_key
from ..tree_builders import get_node_tree
from ..xpath_nodes import XPathNode, DocumentNode, ElementNode
from ..xpath_tokens import XPathFunction, XPathMap, XPathArray
from ..xpath_context import XPathSchemaContext
from ..validators import validate_json_to_xml
from ._xpath31_operators import XPath31Parser
def value_to_etree(v, **attrib):
    if v is None:
        elem = etree.Element(NULL_TAG, **attrib)
    elif isinstance(v, list):
        elem = etree.Element(ARRAY_TAG, **attrib)
        for item in v:
            elem.append(value_to_etree(item))
    elif isinstance(v, bool):
        elem = etree.Element(BOOLEAN_TAG, **attrib)
        elem.text = 'true' if v else 'false'
    elif isinstance(v, (int, float)):
        elem = etree.Element(NUMBER_TAG, **attrib)
        elem.text = str(v)
    elif isinstance(v, str):
        if not escape:
            v = ''.join((x if is_xml_codepoint(ord(x)) else fallback(f'\\u{ord(x):04X}', context=context) for x in v))
            elem = etree.Element(STRING_TAG, **attrib)
        else:
            v = escape_string(v)
            if '\\' in v:
                elem = etree.Element(STRING_TAG, escaped='true', **attrib)
            else:
                elem = etree.Element(STRING_TAG, **attrib)
        elem.text = v
    elif is_etree_element(v):
        v.attrib.update(attrib)
        return v
    else:
        raise ElementPathTypeError(f'unexpected type {type(v)}')
    return elem