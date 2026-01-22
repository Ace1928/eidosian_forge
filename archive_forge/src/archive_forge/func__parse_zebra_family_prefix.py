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
def _parse_zebra_family_prefix(buf):
    """
    Parses family and prefix in Zebra format.
    """
    family, = struct.unpack_from(_ZEBRA_FAMILY_FMT, buf)
    rest = buf[_ZEBRA_FAMILY_SIZE:]
    if socket.AF_INET == family:
        prefix, p_len = struct.unpack_from(_ZEBRA_IPV4_PREFIX_FMT, rest)
        prefix = '%s/%d' % (addrconv.ipv4.bin_to_text(prefix), p_len)
        rest = rest[_ZEBRA_IPV4_PREFIX_SIZE:]
    elif socket.AF_INET6 == family:
        prefix, p_len = struct.unpack_from(_ZEBRA_IPV6_PREFIX_FMT, rest)
        prefix = '%s/%d' % (addrconv.ipv6.bin_to_text(prefix), p_len)
        rest = rest[_ZEBRA_IPV6_PREFIX_SIZE:]
    else:
        raise struct.error('Unsupported family: %d' % family)
    return (family, prefix, rest)