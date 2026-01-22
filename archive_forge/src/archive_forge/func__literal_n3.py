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
def _literal_n3(self, use_plain: bool=False, qname_callback: Optional[Callable[[str], str]]=None) -> str:
    """
        Using plain literal (shorthand) output::
            >>> from rdflib.namespace import XSD

            >>> Literal(1)._literal_n3(use_plain=True)
            '1'

            >>> Literal(1.0)._literal_n3(use_plain=True)
            '1e+00'

            >>> Literal(1.0, datatype=XSD.decimal)._literal_n3(use_plain=True)
            '1.0'

            >>> Literal(1.0, datatype=XSD.float)._literal_n3(use_plain=True)
            '"1.0"^^<http://www.w3.org/2001/XMLSchema#float>'

            >>> Literal("foo", datatype=XSD.string)._literal_n3(
            ...         use_plain=True)
            '"foo"^^<http://www.w3.org/2001/XMLSchema#string>'

            >>> Literal(True)._literal_n3(use_plain=True)
            'true'

            >>> Literal(False)._literal_n3(use_plain=True)
            'false'

            >>> Literal(1.91)._literal_n3(use_plain=True)
            '1.91e+00'

            Only limited precision available for floats:
            >>> Literal(0.123456789)._literal_n3(use_plain=True)
            '1.234568e-01'

            >>> Literal('0.123456789',
            ...     datatype=XSD.decimal)._literal_n3(use_plain=True)
            '0.123456789'

        Using callback for datatype QNames::

            >>> Literal(1)._literal_n3(
            ...         qname_callback=lambda uri: "xsd:integer")
            '"1"^^xsd:integer'

        """
    if use_plain and self.datatype in _PLAIN_LITERAL_TYPES:
        if self.value is not None:
            if self.datatype in _NUMERIC_INF_NAN_LITERAL_TYPES:
                try:
                    v = float(self)
                    if math.isinf(v) or math.isnan(v):
                        return self._literal_n3(False, qname_callback)
                except ValueError:
                    return self._literal_n3(False, qname_callback)
            if self.datatype == _XSD_DOUBLE:
                return sub('\\.?0*e', 'e', '%e' % float(self))
            elif self.datatype == _XSD_DECIMAL:
                s = '%s' % self
                if '.' not in s and 'e' not in s and ('E' not in s):
                    s += '.0'
                return s
            elif self.datatype == _XSD_BOOLEAN:
                return ('%s' % self).lower()
            else:
                return '%s' % self
    encoded = self._quote_encode()
    datatype = self.datatype
    quoted_dt = None
    if datatype is not None:
        if qname_callback:
            quoted_dt = qname_callback(datatype)
        if not quoted_dt:
            quoted_dt = '<%s>' % datatype
        if datatype in _NUMERIC_INF_NAN_LITERAL_TYPES:
            try:
                v = float(self)
                if math.isinf(v):
                    encoded = encoded.replace('inf', 'INF').replace('Infinity', 'INF')
                if math.isnan(v):
                    encoded = encoded.replace('nan', 'NaN')
            except ValueError:
                warnings.warn('Serializing weird numerical %r' % self)
    language = self.language
    if language:
        return '%s@%s' % (encoded, language)
    elif datatype:
        return '%s^^%s' % (encoded, quoted_dt)
    else:
        return '%s' % encoded