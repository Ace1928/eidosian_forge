from copy import deepcopy
from typing import Any, ClassVar, Dict, Optional, Tuple
from ..namespaces import XPATH_MATH_FUNCTIONS_NAMESPACE
from ..datatypes import QName
from ..xpath2 import XPath2Parser

    XPath 3.0 expression parser class. Accepts all XPath 2.0 options as keyword
    arguments, but the *strict* option is ignored because XPath 3.0+ has braced
    URI literals and the expanded name syntax is not compatible.

    :param args: the same positional arguments of class :class:`elementpath.XPath2Parser`.
    :param decimal_formats: a mapping with statically known decimal formats.
    :param defuse_xml: if `True` defuse XML data before parsing, that is the default.
    :param kwargs: the same keyword arguments of class :class:`elementpath.XPath2Parser`.
    