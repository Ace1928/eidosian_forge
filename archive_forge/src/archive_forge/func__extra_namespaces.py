from __future__ import unicode_literals
import sys
import datetime
import sys
import logging
import warnings
import re
import traceback
from . import __author__, __copyright__, __license__, __version__
from .simplexml import SimpleXMLElement, TYPE_MAP, Date, Decimal
@staticmethod
def _extra_namespaces(xml, ns):
    """Extends xml with extra namespaces.
        :param ns: dict with namespaceUrl:prefix pairs
        :param xml: XML node to modify
        """
    if ns:
        _tpl = 'xmlns:%s="%s"'
        _ns_str = ' '.join([_tpl % (prefix, uri) for uri, prefix in ns.items() if uri not in xml])
        xml = xml.replace('/>', ' ' + _ns_str + '/>')
    return xml