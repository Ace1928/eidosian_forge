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
def _perform_storage_service_create(self, path, data):
    request = AzureHTTPRequest()
    request.method = 'POST'
    request.host = AZURE_SERVICE_MANAGEMENT_HOST
    request.path = path
    request.body = data
    request.path, request.query = self._update_request_uri_query(request)
    request.headers = self._update_management_header(request)
    response = self._perform_request(request)
    return response