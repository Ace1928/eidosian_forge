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
@MrtRecord.register_type(MrtRecord.TYPE_TABLE_DUMP)
class TableDumpMrtRecord(MrtCommonRecord):
    """
    MRT Record for the TABLE_DUMP Type.
    """
    MESSAGE_CLS = TableDumpMrtMessage
    SUBTYPE_AFI_IPv4 = 1
    SUBTYPE_AFI_IPv6 = 2