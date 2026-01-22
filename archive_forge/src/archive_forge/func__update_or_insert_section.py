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
def _update_or_insert_section(self, res, section, prev_section, text):
    try:
        res.object.find(fixxpath(res.object, section)).text = text
    except Exception:
        for i, e in enumerate(res.object):
            tag = '{http://www.vmware.com/vcloud/v1.5}%s' % prev_section
            if e.tag == tag:
                break
        e = ET.Element('{http://www.vmware.com/vcloud/v1.5}%s' % section)
        e.text = text
        res.object.insert(i, e)
    return res