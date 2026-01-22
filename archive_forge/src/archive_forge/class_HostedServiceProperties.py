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
class HostedServiceProperties(WindowsAzureData):

    def __init__(self):
        self.description = ''
        self.location = ''
        self.affinity_group = ''
        self.label = _Base64String()
        self.status = ''
        self.date_created = ''
        self.date_last_modified = ''
        self.extended_properties = _DictOf('ExtendedProperty', 'Name', 'Value')