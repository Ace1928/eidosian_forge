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
def _construct_element_tree(self, source, etree):
    if source is None:
        return ET.Element()
    if isinstance(source, list):
        for value in source:
            etree.append(self._construct_element_tree(value, etree))
    elif isinstance(source, WindowsAzureData):
        class_name = source.__class__.__name__
        etree.append(ET.Element(class_name))
        for name, value in vars(source).items():
            if value is not None:
                if isinstance(value, list) or isinstance(value, WindowsAzureData):
                    etree.append(self._construct_element_tree(value, etree))
                else:
                    ele = ET.Element(self._get_serialization_name(name))
                    ele.text = xml_escape(str(value))
                    etree.append(ele)
        etree.append(ET.Element(class_name))
    return etree