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
def _fill_instance_child(self, xmldoc, element_name, return_type):
    """
        Converts a child of the current dom element to the specified type.
        """
    xmlelements = self._get_child_nodes(xmldoc, self._get_serialization_name(element_name))
    if not xmlelements:
        return None
    return_obj = return_type()
    self._fill_data_to_return_object(xmlelements[0], return_obj)
    return return_obj