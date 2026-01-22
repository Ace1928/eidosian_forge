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
def _update_management_header(self, request):
    """
        Add additional headers for management.
        """
    if request.method in ['PUT', 'POST', 'MERGE', 'DELETE']:
        request.headers['Content-Length'] = str(len(request.body))
    if request.method not in ['GET', 'HEAD']:
        for key in request.headers:
            if 'content-type' == key.lower():
                break
        else:
            request.headers['Content-Type'] = 'application/xml'
    return request.headers