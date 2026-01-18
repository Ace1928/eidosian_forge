import struct
from os_ken.lib.pack_utils import msg_pack_into
from . import packet_base
from . import packet_utils
from . import ether_types

    Generate instance of GRE class with information for NVGRE (RFC7637).

    :param version: Version.
    :param vsid: Virtual Subnet ID.
    :param flow_id: FlowID.
    :return: Instance of GRE class with information for NVGRE.
    