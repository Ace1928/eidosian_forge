import re
import copy
import time
import base64
import random
import collections
from xml.dom import minidom
from datetime import datetime
from xml.sax.saxutils import escape as xml_escape
from libcloud.utils.py3 import ET, httplib, urlparse
from libcloud.utils.py3 import urlquote as url_quote
from libcloud.utils.py3 import _real_unicode, ensure_string
from libcloud.utils.misc import ReprMixin
from libcloud.common.azure import AzureRedirectException, AzureServiceManagementConnection
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.compute.types import NodeState
from libcloud.compute.providers import Provider
def _fill_dict_of(self, xmldoc, parent_xml_element_name, pair_xml_element_name, key_xml_element_name, value_xml_element_name):
    return_obj = {}
    xmlelements = self._get_child_nodes(xmldoc, parent_xml_element_name)
    if xmlelements:
        xmlelements = self._get_child_nodes(xmlelements[0], pair_xml_element_name)
        for pair in xmlelements:
            keys = self._get_child_nodes(pair, key_xml_element_name)
            values = self._get_child_nodes(pair, value_xml_element_name)
            if keys and values:
                key = keys[0].firstChild.nodeValue
                value = values[0].firstChild.nodeValue
                return_obj[key] = value
    return return_obj