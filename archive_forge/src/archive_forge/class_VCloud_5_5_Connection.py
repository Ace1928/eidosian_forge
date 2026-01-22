import os
import re
import copy
import time
import base64
import datetime
from xml.parsers.expat import ExpatError
from libcloud.utils.py3 import ET, b, next, httplib, urlparse, urlencode
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
class VCloud_5_5_Connection(VCloud_1_5_Connection):

    def _get_auth_headers(self):
        """Compatibility for using v5.5 of the API"""
        auth_headers = super()._get_auth_headers()
        auth_headers['Accept'] = 'application/*+xml;version=5.5'
        return auth_headers

    def add_default_headers(self, headers):
        headers['Accept'] = 'application/*+xml;version=5.5'
        headers['x-vcloud-authorization'] = self.token
        return headers