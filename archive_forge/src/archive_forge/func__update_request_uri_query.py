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
def _update_request_uri_query(self, request):
    """
        pulls the query string out of the URI and moves it into
        the query portion of the request object.  If there are already
        query parameters on the request the parameters in the URI will
        appear after the existing parameters
        """
    if '?' in request.path:
        request.path, _, query_string = request.path.partition('?')
        if query_string:
            query_params = query_string.split('&')
            for query in query_params:
                if '=' in query:
                    name, _, value = query.partition('=')
                    request.query.append((name, value))
    request.path = url_quote(request.path, "/()$=',")
    if request.query:
        request.path += '?'
        for name, value in request.query:
            if value is not None:
                request.path += '{}={}{}'.format(name, url_quote(value, "/()$=',"), '&')
        request.path = request.path[:-1]
    return (request.path, request.query)