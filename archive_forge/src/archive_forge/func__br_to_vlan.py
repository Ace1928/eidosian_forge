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
def _br_to_vlan(self, ctx, br_name):
    ctx.populate_cache()
    br = ctx.find_bridge(br_name, must_exist=True)
    vlan = br.vlan
    if isinstance(vlan, list):
        if len(vlan) == 0:
            vlan = 0
        else:
            vlan = vlan[0]
    return vlan