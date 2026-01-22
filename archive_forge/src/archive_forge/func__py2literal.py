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
def _py2literal(obj: Any, pType: Any, castFunc: Optional[Callable[[Any], Any]], dType: Optional[_StrT]) -> Tuple[Any, Optional[_StrT]]:
    if castFunc is not None:
        return (castFunc(obj), dType)
    elif dType is not None:
        return (obj, dType)
    else:
        return (obj, None)