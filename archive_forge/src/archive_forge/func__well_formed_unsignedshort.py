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
def _well_formed_unsignedshort(lexical: Union[str, bytes], value: Any) -> bool:
    """
    xsd:unsignedShort has a 16bit value of between 0 and 65535
    """
    return len(lexical) > 0 and isinstance(value, int) and (0 <= value <= 65535)