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
def _list_ifaces_verbose(self, ctx, datapath_id, port_name):
    ctx.populate_cache()
    br = ctx.find_bridge_by_id(datapath_id, True)
    ctx.verify_ports()
    iface_cfgs = []
    if port_name is None:
        for vsctl_port in br.ports:
            iface_cfgs.extend((self._iface_to_dict(vsctl_iface.iface_cfg) for vsctl_iface in vsctl_port.ifaces))
    else:
        for vsctl_port in br.ports:
            iface_cfgs.extend((self._iface_to_dict(vsctl_iface.iface_cfg) for vsctl_iface in vsctl_port.ifaces if vsctl_iface.iface_cfg.name == port_name))
    return iface_cfgs