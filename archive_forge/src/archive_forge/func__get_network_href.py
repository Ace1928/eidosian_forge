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
def _get_network_href(self, network_name):
    network_href = None
    res = self.connection.request(self.org)
    links = res.object.findall(fixxpath(res.object, 'Link'))
    for link in links:
        if link.attrib['type'] == 'application/vnd.vmware.vcloud.orgNetwork+xml' and link.attrib['name'] == network_name:
            network_href = link.attrib['href']
    if network_href is None:
        raise ValueError('%s is not a valid organisation network name' % network_name)
    else:
        return network_href