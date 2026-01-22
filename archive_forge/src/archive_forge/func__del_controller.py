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
def _del_controller(self, ctx, br_name):
    ctx.populate_cache()
    br = ctx.find_real_bridge(br_name, True)
    ovsrec_bridge = br.br_cfg
    self._verify_controllers(ovsrec_bridge)
    if ovsrec_bridge.controller:
        self._delete_controllers(ovsrec_bridge.controller)
        ovsrec_bridge.controller = []