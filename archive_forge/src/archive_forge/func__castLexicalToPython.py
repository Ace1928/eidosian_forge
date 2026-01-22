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
def _castLexicalToPython(lexical: Union[str, bytes], datatype: Optional[URIRef]) -> Any:
    """
    Map a lexical form to the value-space for the given datatype
    :returns: a python object for the value or ``None``
    """
    try:
        conv_func = _toPythonMapping[datatype]
    except KeyError:
        return None
    if conv_func is not None:
        try:
            return conv_func(lexical)
        except Exception:
            logger.warning('Failed to convert Literal lexical form to value. Datatype=%s, Converter=%s', datatype, conv_func, exc_info=True)
            return None
    else:
        try:
            return str(lexical)
        except UnicodeDecodeError:
            return str(lexical, 'utf-8')