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
def _neutron_attach_ip_to_server(self, server, floating_ip, fixed_address=None, nat_destination=None):
    port, fixed_address = self._nat_destination_port(server, fixed_address=fixed_address, nat_destination=nat_destination)
    if not port:
        raise exceptions.SDKException('unable to find a port for server {0}'.format(server['id']))
    floating_ip_args = {'port_id': port['id']}
    if fixed_address is not None:
        floating_ip_args['fixed_ip_address'] = fixed_address
    return self.network.update_ip(floating_ip, **floating_ip_args)