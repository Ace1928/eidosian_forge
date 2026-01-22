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
def _neutron_create_floating_ip(self, network_name_or_id=None, server=None, fixed_address=None, nat_destination=None, port=None, wait=False, timeout=60, network_id=None):
    if not network_id:
        if network_name_or_id:
            try:
                network = self.network.find_network(network_name_or_id)
            except exceptions.ResourceNotFound:
                raise exceptions.NotFoundException('unable to find network for floating ips with ID {0}'.format(network_name_or_id))
            network_id = network['id']
        else:
            network_id = self._get_floating_network_id()
    kwargs = {'floating_network_id': network_id}
    if not port:
        if server:
            port_obj, fixed_ip_address = self._nat_destination_port(server, fixed_address=fixed_address, nat_destination=nat_destination)
            if port_obj:
                port = port_obj['id']
            if fixed_ip_address:
                kwargs['fixed_ip_address'] = fixed_ip_address
    if port:
        kwargs['port_id'] = port
    fip = self._submit_create_fip(kwargs)
    fip_id = fip['id']
    if port:
        if wait:
            try:
                for count in utils.iterate_timeout(timeout, 'Timeout waiting for the floating IP to be ACTIVE', wait=min(5, timeout)):
                    fip = self.get_floating_ip(fip_id)
                    if fip and fip['status'] == 'ACTIVE':
                        break
            except exceptions.ResourceTimeout:
                self.log.error('Timed out on floating ip %(fip)s becoming active. Deleting', {'fip': fip_id})
                try:
                    self.delete_floating_ip(fip_id)
                except Exception as e:
                    self.log.error('FIP LEAK: Attempted to delete floating ip %(fip)s but received %(exc)s exception: %(err)s', {'fip': fip_id, 'exc': e.__class__, 'err': str(e)})
                raise
        if fip['port_id'] != port:
            if server:
                raise exceptions.SDKException('Attempted to create FIP on port {port} for server {server} but FIP has port {port_id}'.format(port=port, port_id=fip['port_id'], server=server['id']))
            else:
                raise exceptions.SDKException('Attempted to create FIP on port {port} but something went wrong'.format(port=port))
    return fip