from __future__ import annotations
from typing import TYPE_CHECKING
from ..core import BaseDomain
class NetworkSubnet(BaseDomain):
    """Network Subnet Domain

    :param type: str
              Type of sub network.
    :param ip_range: str
              Range to allocate IPs from.
    :param network_zone: str
              Name of network zone.
    :param gateway: str
              Gateway for the route.
    :param vswitch_id: int
              ID of the vSwitch.
    """
    TYPE_SERVER = 'server'
    'Subnet Type server, deprecated, use TYPE_CLOUD instead'
    TYPE_CLOUD = 'cloud'
    'Subnet Type cloud'
    TYPE_VSWITCH = 'vswitch'
    'Subnet Type vSwitch'
    __slots__ = ('type', 'ip_range', 'network_zone', 'gateway', 'vswitch_id')

    def __init__(self, ip_range: str, type: str | None=None, network_zone: str | None=None, gateway: str | None=None, vswitch_id: int | None=None):
        self.type = type
        self.ip_range = ip_range
        self.network_zone = network_zone
        self.gateway = gateway
        self.vswitch_id = vswitch_id