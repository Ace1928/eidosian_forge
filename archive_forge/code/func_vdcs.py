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
@property
def vdcs(self):
    """
        vCloud virtual data centers (vDCs).

        :return: list of vDC objects
        :rtype: ``list`` of :class:`Vdc`
        """
    if not self._vdcs:
        self.connection.check_org()
        res = self.connection.request(self.org)
        self._vdcs = [self._to_vdc(self.connection.request(get_url_path(i.get('href'))).object) for i in res.object.findall(fixxpath(res.object, 'Link')) if i.get('type') == 'application/vnd.vmware.vcloud.vdc+xml']
    return self._vdcs