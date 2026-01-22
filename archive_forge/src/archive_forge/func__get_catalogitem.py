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
def _get_catalogitem(self, catalog_item):
    """Given a catalog item href returns elementree"""
    res = self.connection.request(get_url_path(catalog_item), headers={'Content-Type': 'application/vnd.vmware.vcloud.catalogItem+xml'}).object
    return res