import http.client as http
import os
import sys
import urllib.parse as urlparse
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from oslo_utils import uuidutils
from webob import exc
from glance.common import config
from glance.common import exception
from glance.common import utils
from glance.i18n import _, _LE, _LI, _LW
@staticmethod
def _header_list_to_dict(headers):
    """Expand a list of headers into a dictionary.

        headers: a list of [(key, value), (key, value), (key, value)]

        Returns: a dictionary representation of the list
        """
    d = {}
    for header, value in headers:
        if header.startswith('x-image-meta-property-'):
            prop = header.replace('x-image-meta-property-', '')
            d.setdefault('properties', {})
            d['properties'][prop] = value
        else:
            d[header.replace('x-image-meta-', '')] = value
    return d