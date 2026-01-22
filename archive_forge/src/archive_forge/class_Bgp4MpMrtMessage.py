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
class Bgp4MpMrtMessage(MrtMessage, metaclass=abc.ABCMeta):
    """
    MRT Message for the BGP4MP Type.
    """
    _TYPE = {'ascii': ['peer_ip', 'local_ip']}