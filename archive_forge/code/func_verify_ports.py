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
def verify_ports(self):
    if self.verified_ports:
        return
    self.verify_bridges()
    for ovsrec_bridge in self.idl.tables[vswitch_idl.OVSREC_TABLE_BRIDGE].rows.values():
        ovsrec_bridge.verify(vswitch_idl.OVSREC_BRIDGE_COL_PORTS)
    for ovsrec_port in self.idl.tables[vswitch_idl.OVSREC_TABLE_PORT].rows.values():
        ovsrec_port.verify(vswitch_idl.OVSREC_PORT_COL_INTERFACES)
    self.verified_ports = True