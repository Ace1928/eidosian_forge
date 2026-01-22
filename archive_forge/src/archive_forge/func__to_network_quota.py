import base64
import warnings
from libcloud.pricing import get_size_price
from libcloud.utils.py3 import ET, b, next, httplib, parse_qs, urlparse
from libcloud.utils.xml import findall
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.iso8601 import parse_date
from libcloud.common.openstack import (
from libcloud.utils.networking import is_public_subnet
from libcloud.common.exceptions import BaseHTTPError
def _to_network_quota(self, obj):
    res = OpenStack_2_NetworkQuota(floatingip=obj['floatingip'], network=obj['network'], port=obj['port'], rbac_policy=obj['rbac_policy'], router=obj.get('router', None), security_group=obj.get('security_group', None), security_group_rule=obj.get('security_group_rule', None), subnet=obj.get('subnet', None), subnetpool=obj.get('subnetpool', None), driver=self.connection.driver)
    return res