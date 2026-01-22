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
def add_port_to_cache(self, vsctl_bridge_parent, ovsrec_port):
    tag = getattr(ovsrec_port, vswitch_idl.OVSREC_PORT_COL_TAG, None)
    if isinstance(tag, list):
        if len(tag) == 0:
            tag = 0
        else:
            tag = tag[0]
    if tag is not None and 0 <= tag < 4096:
        vlan_bridge = vsctl_bridge_parent.find_vlan_bridge(tag)
        if vlan_bridge:
            vsctl_bridge_parent = vlan_bridge
    vsctl_port = VSCtlPort(vsctl_bridge_parent, ovsrec_port)
    vsctl_bridge_parent.ports.add(vsctl_port)
    self.ports[ovsrec_port.name] = vsctl_port
    return vsctl_port