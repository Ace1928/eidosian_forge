import contextlib
import socket
import struct
from os_ken.controller import handler
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.lib import addrconv
from os_ken.lib import hub
from os_ken.lib.packet import arp
from os_ken.lib.packet import vrrp
from os_ken.services.protocols.vrrp import monitor
from os_ken.services.protocols.vrrp import event as vrrp_event
from os_ken.services.protocols.vrrp import utils

    This module uses raw socket so that privilege(CAP_NET_ADMIN capability)
    is required.
    