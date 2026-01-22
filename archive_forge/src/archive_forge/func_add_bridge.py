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
def add_bridge(self, br_name, parent_name=None, vlan=0, may_exist=False):
    self.populate_cache()
    if may_exist:
        vsctl_bridge = self.find_bridge(br_name, False)
        if vsctl_bridge:
            if not parent_name:
                if vsctl_bridge.parent:
                    vsctl_fatal('"--may-exist add-vsctl_bridge %s" but %s is a VLAN bridge for VLAN %d' % (br_name, br_name, vsctl_bridge.vlan))
            elif not vsctl_bridge.parent:
                vsctl_fatal('"--may-exist add-vsctl_bridge %s %s %d" but %s is not a VLAN bridge' % (br_name, parent_name, vlan, br_name))
            elif vsctl_bridge.parent.name != parent_name:
                vsctl_fatal('"--may-exist add-vsctl_bridge %s %s %d" but %s has the wrong parent %s' % (br_name, parent_name, vlan, br_name, vsctl_bridge.parent.name))
            elif vsctl_bridge.vlan != vlan:
                vsctl_fatal('"--may-exist add-vsctl_bridge %s %s %d" but %s is a VLAN bridge for the wrong VLAN %d' % (br_name, parent_name, vlan, br_name, vsctl_bridge.vlan))
            return
    self.check_conflicts(br_name, 'cannot create a bridge named %s' % br_name)
    txn = self.txn
    tables = self.idl.tables
    if not parent_name:
        ovsrec_iface = txn.insert(tables[vswitch_idl.OVSREC_TABLE_INTERFACE])
        ovsrec_iface.name = br_name
        ovsrec_iface.type = 'internal'
        ovsrec_port = txn.insert(tables[vswitch_idl.OVSREC_TABLE_PORT])
        ovsrec_port.name = br_name
        ovsrec_port.interfaces = [ovsrec_iface]
        ovsrec_port.fake_bridge = False
        ovsrec_bridge = txn.insert(tables[vswitch_idl.OVSREC_TABLE_BRIDGE])
        ovsrec_bridge.name = br_name
        ovsrec_bridge.ports = [ovsrec_port]
        self.ovs_insert_bridge(ovsrec_bridge)
    else:
        parent = self.find_bridge(parent_name, False)
        if parent and parent.parent:
            vsctl_fatal('cannot create bridge with fake bridge as parent')
        if not parent:
            vsctl_fatal('parent bridge %s does not exist' % parent_name)
        ovsrec_iface = txn.insert(tables[vswitch_idl.OVSREC_TABLE_INTERFACE])
        ovsrec_iface.name = br_name
        ovsrec_iface.type = 'internal'
        ovsrec_port = txn.insert(tables[vswitch_idl.OVSREC_TABLE_PORT])
        ovsrec_port.name = br_name
        ovsrec_port.interfaces = [ovsrec_iface]
        ovsrec_port.fake_bridge = True
        ovsrec_port.tag = vlan
        self.bridge_insert_port(parent.br_cfg, ovsrec_port)
    self.invalidate_cache()