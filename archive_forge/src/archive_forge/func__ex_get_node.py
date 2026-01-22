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
def _ex_get_node(self, node_id):
    """
        Get a node instance from a node ID.

        :param node_id: ID of the node
        :type node_id: ``str``

        :return: node instance or None if not found
        :rtype: :class:`Node` or ``None``
        """
    res = self.connection.request(get_url_path(node_id), headers={'Content-Type': 'application/vnd.vmware.vcloud.vApp+xml'})
    return self._to_node(res.object)