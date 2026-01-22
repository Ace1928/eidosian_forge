import re
import xml.etree.ElementTree as etree
from io import BytesIO
from copy import deepcopy
from time import sleep
from base64 import b64encode
from typing import Dict
from functools import wraps
from libcloud.utils.py3 import b, httplib, basestring
from libcloud.utils.xml import findtext
from libcloud.common.base import RawResponse, XmlResponse, ConnectionUserAndKey
from libcloud.compute.base import Node
from libcloud.compute.types import LibcloudError, InvalidCredsError
class NttCisNatRule:
    """
    An IP NAT rule in a network domain
    """

    def __init__(self, id, network_domain, internal_ip, external_ip, status):
        self.id = id
        self.network_domain = network_domain
        self.internal_ip = internal_ip
        self.external_ip = external_ip
        self.status = status

    def __repr__(self):
        return '<NttCisNatRule: id=%s, status=%s>' % (self.id, self.status)