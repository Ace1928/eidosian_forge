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
def _nova_delete_floating_ip(self, floating_ip_id):
    try:
        proxy._json_response(self.compute.delete('/os-floating-ips/{id}'.format(id=floating_ip_id)), error_message='Unable to delete floating IP {fip_id}'.format(fip_id=floating_ip_id))
    except exceptions.NotFoundException:
        return False
    return True