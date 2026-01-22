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
class HostingComConnection(VCloudConnection):
    """
    vCloud connection subclass for Hosting.com
    """
    host = 'vcloud.safesecureweb.com'

    def _get_auth_headers(self):
        """hosting.com doesn't follow the standard vCloud authentication API"""
        return {'Authentication': base64.b64encode(b('{}:{}'.format(self.user_id, self.key))), 'Content-Length': '0'}