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
def _perform_power_operation(self, node, operation):
    res = self.connection.request('{}/power/action/{}'.format(get_url_path(node.id), operation), method='POST')
    self._wait_for_task_completion(res.object.get('href'))
    res = self.connection.request(get_url_path(node.id))
    return self._to_node(res.object)