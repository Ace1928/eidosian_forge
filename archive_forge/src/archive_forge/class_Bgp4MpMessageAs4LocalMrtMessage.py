import abc
import logging
import struct
import time
from os_ken.lib import addrconv
from os_ken.lib import ip
from os_ken.lib import stringify
from os_ken.lib import type_desc
from os_ken.lib.packet import bgp
from os_ken.lib.packet import ospf
@Bgp4MpMrtMessage.register_type(Bgp4MpMrtRecord.SUBTYPE_BGP4MP_MESSAGE_AS4_LOCAL)
class Bgp4MpMessageAs4LocalMrtMessage(Bgp4MpMessageAs4MrtMessage):
    """
    MRT Message for the BGP4MP Type and the BGP4MP_MESSAGE_AS4_LOCAL subtype.
    """