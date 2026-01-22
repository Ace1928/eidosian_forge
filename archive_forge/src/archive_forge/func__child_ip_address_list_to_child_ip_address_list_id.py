import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
@staticmethod
def _child_ip_address_list_to_child_ip_address_list_id(child_ip_addr_list):
    return dd_object_to_id(child_ip_addr_list, NttCisChildIpAddressList, id_value='id')