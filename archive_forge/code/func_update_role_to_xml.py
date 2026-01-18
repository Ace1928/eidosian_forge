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
@staticmethod
def update_role_to_xml(role_name, os_virtual_hard_disk, role_type, network_configuration_set, availability_set_name, data_virtual_hard_disks, vm_image_name, role_size):
    doc = AzureXmlSerializer.doc_from_xml('PersistentVMRole')
    AzureXmlSerializer.role_to_xml(availability_set_name, data_virtual_hard_disks, network_configuration_set, os_virtual_hard_disk, vm_image_name, role_name, role_size, role_type, None, doc)
    result = ensure_string(ET.tostring(doc, encoding='utf-8'))
    return result