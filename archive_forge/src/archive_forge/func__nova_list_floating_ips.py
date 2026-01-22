import ipaddress
import time
import warnings
from openstack.cloud import _utils
from openstack.cloud import exc
from openstack.cloud import meta
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
from openstack import proxy
from openstack import utils
from openstack import warnings as os_warnings
def _nova_list_floating_ips(self):
    try:
        data = proxy._json_response(self.compute.get('/os-floating-ips'))
    except exceptions.NotFoundException:
        return []
    return self._get_and_munchify('floating_ips', data)