import abc
import socket
import struct
import logging
import netaddr
from packaging import version as packaging_version
from os_ken import flags as cfg_flags  # For loading 'zapi' option definition
from os_ken.cfg import CONF
from os_ken.lib import addrconv
from os_ken.lib import ip
from os_ken.lib import stringify
from os_ken.lib import type_desc
from . import packet_base
from . import bgp
from . import safi as packet_safi
@_FrrZebraMessageBody.register_type(FRR_ZEBRA_ROUTER_ID_UPDATE)
@_ZebraMessageBody.register_type(ZEBRA_ROUTER_ID_UPDATE)
class ZebraRouterIDUpdate(_ZebraMessageBody):
    """
    Message body class for ZEBRA_ROUTER_ID_UPDATE.
    """

    def __init__(self, family, prefix):
        super(ZebraRouterIDUpdate, self).__init__()
        self.family = family
        if isinstance(prefix, (IPv4Prefix, IPv6Prefix)):
            prefix = prefix.prefix
        self.prefix = prefix

    @classmethod
    def parse(cls, buf, version=_DEFAULT_VERSION):
        family, prefix, _ = _parse_zebra_family_prefix(buf)
        return cls(family, prefix)

    def serialize(self, version=_DEFAULT_VERSION):
        self.family, buf = _serialize_zebra_family_prefix(self.prefix)
        return buf