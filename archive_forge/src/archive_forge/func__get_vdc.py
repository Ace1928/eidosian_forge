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
def _get_vdc(self, vdc_name):
    vdc = None
    if not vdc_name:
        vdc = self.vdcs[0]
    else:
        for v in self.vdcs:
            if v.name == vdc_name or v.id == vdc_name:
                vdc = v
        if vdc is None:
            raise ValueError('%s virtual data centre could not be found' % vdc_name)
    return vdc