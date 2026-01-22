import ipaddress
import socket
from openstack import _log
from openstack import exceptions
from openstack import utils
def _get_interface_ip(cloud, server):
    """Get the interface IP for the server

    Interface IP is the IP that should be used for communicating with the
    server. It is:
    - the IP on the configured default_interface network
    - if cloud.private, the private ip if it exists
    - if the server has a public ip, the public ip
    """
    default_ip = get_server_default_ip(cloud, server)
    if default_ip:
        return default_ip
    if cloud.private and server['private_v4']:
        return server['private_v4']
    if server['public_v6'] and cloud._local_ipv6 and (not cloud.force_ipv4):
        return server['public_v6']
    else:
        return server['public_v4']