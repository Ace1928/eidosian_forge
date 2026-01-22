import re
from fractions import Fraction
import logging
import math
import warnings
import xml.dom.minidom
from base64 import b64decode, b64encode
from binascii import hexlify, unhexlify
from collections import defaultdict
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from re import compile, sub
from typing import (
from urllib.parse import urldefrag, urljoin, urlparse
from isodate import (
import rdflib
import rdflib.util
from rdflib.compat import long_type
def _castPythonToLiteral(obj: Any, datatype: Optional[str]) -> Tuple[Any, Optional[str]]:
    """
    Casts a tuple of a python type and a special datatype URI to a tuple of the lexical value and a
    datatype URI (or None)
    """
    castFunc: Optional[Callable[[Any], Union[str, bytes]]]
    dType: Optional[str]
    for (pType, dType), castFunc in _SpecificPythonToXSDRules:
        if isinstance(obj, pType) and dType == datatype:
            return _py2literal(obj, pType, castFunc, dType)
    for pType, (castFunc, dType) in _GenericPythonToXSDRules:
        if isinstance(obj, pType):
            return _py2literal(obj, pType, castFunc, dType)
    return (obj, None)