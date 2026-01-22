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
def _verify_controllers(self, ovsrec_bridge):
    ovsrec_bridge.verify(vswitch_idl.OVSREC_BRIDGE_COL_CONTROLLER)
    for controller in ovsrec_bridge.controller:
        controller.verify(vswitch_idl.OVSREC_CONTROLLER_COL_TARGET)