import ipaddress
import socket
from openstack import _log
from openstack import exceptions
from openstack import utils
def _get_supplemental_addresses(cloud, server):
    fixed_ip_mapping = {}
    for name, network in server['addresses'].items():
        for address in network:
            if address['version'] == 6:
                continue
            if address.get('OS-EXT-IPS:type') == 'floating':
                return server['addresses']
            fixed_ip_mapping[address['addr']] = name
    try:
        if cloud.has_service('network') and cloud._has_floating_ips() and (server['status'] == 'ACTIVE'):
            for port in cloud.search_ports(filters=dict(device_id=server['id'])):
                for fip in cloud.search_floating_ips(filters=dict(port_id=port['id'])):
                    fixed_net = fixed_ip_mapping.get(fip['fixed_ip_address'])
                    if fixed_net is None:
                        log = _log.setup_logging('openstack')
                        log.debug('The cloud returned floating ip %(fip)s attached to server %(server)s but the fixed ip associated with the floating ip in the neutron listing does not exist in the nova listing. Something is exceptionally broken.', dict(fip=fip['id'], server=server['id']))
                    else:
                        server['addresses'][fixed_net].append(_make_address_dict(fip, port))
    except exceptions.SDKException:
        pass
    return server['addresses']