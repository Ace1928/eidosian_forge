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
class VSCtlBridge(object):

    def __init__(self, ovsrec_bridge, name, parent, vlan):
        super(VSCtlBridge, self).__init__()
        self.br_cfg = ovsrec_bridge
        self.name = name
        self.ports = set()
        self.parent = parent
        self.vlan = vlan
        self.children = set()

    def find_vlan_bridge(self, vlan):
        return ifind(lambda child: child.vlan == vlan, self.children)