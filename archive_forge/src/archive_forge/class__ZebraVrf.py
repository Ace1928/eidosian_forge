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
class _ZebraVrf(_ZebraMessageBody):
    """
    Base class for FRR_ZEBRA_VRF_ADD and FRR_ZEBRA_VRF_DELETE message body.
    """
    _HEADER_FMT = '!%ds' % VRF_NAMSIZ

    def __init__(self, vrf_name):
        super(_ZebraVrf, self).__init__()
        self.vrf_name = vrf_name

    @classmethod
    def parse(cls, buf, version=_DEFAULT_FRR_VERSION):
        vrf_name_bin = buf[:VRF_NAMSIZ]
        vrf_name = str(str(vrf_name_bin.strip(b'\x00'), 'ascii'))
        return cls(vrf_name)

    def serialize(self, version=_DEFAULT_FRR_VERSION):
        return struct.pack(self._HEADER_FMT, self.vrf_name.encode('ascii'))