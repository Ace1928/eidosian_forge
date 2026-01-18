import logging
import operator
import os
import re
import sys
import weakref
import ovs.db.data
import ovs.db.parser
import ovs.db.schema
import ovs.db.types
import ovs.poller
import ovs.json
from ovs import jsonrpc
from ovs import ovsuuid
from ovs import stream
from ovs.db import idl
from os_ken.lib import hub
from os_ken.lib import ip
from os_ken.lib.ovs import vswitch_idl
from os_ken.lib.stringify import StringifyMixin
def valid_ovsdb_addr(addr):
    """
    Returns True if the given addr is valid OVSDB server address, otherwise
    False.

    The valid formats are:

    - ``unix:file``
    - ``tcp:ip:port``
    - ``ssl:ip:port``

    If ip is IPv6 address, wrap ip with brackets (e.g., ssl:[::1]:6640).

    :param addr: str value of OVSDB server address.
    :return: True if valid, otherwise False.
    """
    m = re.match('unix:(\\S+)', addr)
    if m:
        file = m.group(1)
        return os.path.isfile(file)
    m = re.match('(tcp|ssl):(\\S+):(\\d+)', addr)
    if m:
        address = m.group(2)
        port = m.group(3)
        if '[' in address:
            address = address.strip('[').strip(']')
            return ip.valid_ipv6(address) and port.isdigit()
        else:
            return ip.valid_ipv4(address) and port.isdigit()
    return False