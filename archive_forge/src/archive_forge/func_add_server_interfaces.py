import ipaddress
import socket
from openstack import _log
from openstack import exceptions
from openstack import utils
def add_server_interfaces(cloud, server):
    """Add network interface information to server.

    Query the cloud as necessary to add information to the server record
    about the network information needed to interface with the server.

    Ensures that public_v4, public_v6, private_v4, private_v6, interface_ip,
                 accessIPv4 and accessIPv6 are always set.
    """
    server['addresses'] = _get_supplemental_addresses(cloud, server)
    server['public_v4'] = get_server_external_ipv4(cloud, server) or ''
    if cloud.force_ipv4:
        server['public_v6'] = ''
    else:
        server['public_v6'] = get_server_external_ipv6(server) or ''
    server['private_v4'] = get_server_private_ip(server, cloud) or ''
    server['interface_ip'] = _get_interface_ip(cloud, server) or ''
    if cloud.private and server.private_v4:
        server['access_ipv4'] = server['private_v4']
    else:
        server['access_ipv4'] = server['public_v4']
    server['access_ipv6'] = server['public_v6']
    return server