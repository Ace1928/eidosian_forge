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
def _is_storage_service_unique(self, service_name=None):
    if not service_name:
        raise ValueError('service_name is required.')
    _check_availability = self._perform_get('%s/operations/isavailable/%s%s' % (self._get_storage_service_path(), _str(service_name), ''), AvailabilityResponse)
    self.raise_for_response(_check_availability, 200)
    return _check_availability.result