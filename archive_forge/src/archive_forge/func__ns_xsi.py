import datetime
import logging
from lxml import etree
import io
import warnings
import prov
import prov.identifier
from prov.model import DEFAULT_NAMESPACES, sorted_attributes
from prov.constants import *  # NOQA
from prov.serializers import Serializer
def _ns_xsi(tag):
    return _ns(DEFAULT_NAMESPACES['xsi'].uri, tag)