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
def _nova_create_floating_ip(self, pool=None):
    with _utils.openstacksdk_exceptions('Unable to create floating IP in pool {pool}'.format(pool=pool)):
        if pool is None:
            pools = self.list_floating_ip_pools()
            if not pools:
                raise exceptions.NotFoundException('unable to find a floating ip pool')
            pool = pools[0]['name']
        data = proxy._json_response(self.compute.post('/os-floating-ips', json=dict(pool=pool)))
        pool_ip = self._get_and_munchify('floating_ip', data)
        data = proxy._json_response(self.compute.get('/os-floating-ips/{id}'.format(id=pool_ip['id'])))
        return self._get_and_munchify('floating_ip', data)