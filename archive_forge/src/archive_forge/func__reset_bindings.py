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
def _reset_bindings() -> None:
    """
    Reset lexical<->value space binding for `Literal`
    """
    _toPythonMapping.clear()
    _toPythonMapping.update(XSDToPython)
    _GenericPythonToXSDRules.clear()
    _GenericPythonToXSDRules.extend(_OriginalGenericPythonToXSDRules)
    _SpecificPythonToXSDRules.clear()
    _SpecificPythonToXSDRules.extend(_OriginalSpecificPythonToXSDRules)