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
def _delete_floating_ip(self, floating_ip_id):
    if self._use_neutron_floating():
        try:
            return self._neutron_delete_floating_ip(floating_ip_id)
        except exceptions.NotFoundException as e:
            self.log.debug("Something went wrong talking to neutron API: '%(msg)s'. Trying with Nova.", {'msg': str(e)})
    return self._nova_delete_floating_ip(floating_ip_id)