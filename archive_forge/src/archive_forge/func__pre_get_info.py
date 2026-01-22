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
def _pre_get_info(self, _ctx, _command):
    schema_helper = self.schema_helper
    schema_helper.register_columns(vswitch_idl.OVSREC_TABLE_OPEN_VSWITCH, [vswitch_idl.OVSREC_OPEN_VSWITCH_COL_BRIDGES])
    schema_helper.register_columns(vswitch_idl.OVSREC_TABLE_BRIDGE, [vswitch_idl.OVSREC_BRIDGE_COL_NAME, vswitch_idl.OVSREC_BRIDGE_COL_CONTROLLER, vswitch_idl.OVSREC_BRIDGE_COL_FAIL_MODE, vswitch_idl.OVSREC_BRIDGE_COL_PORTS])
    schema_helper.register_columns(vswitch_idl.OVSREC_TABLE_PORT, [vswitch_idl.OVSREC_PORT_COL_NAME, vswitch_idl.OVSREC_PORT_COL_FAKE_BRIDGE, vswitch_idl.OVSREC_PORT_COL_TAG, vswitch_idl.OVSREC_PORT_COL_INTERFACES, vswitch_idl.OVSREC_PORT_COL_QOS])
    schema_helper.register_columns(vswitch_idl.OVSREC_TABLE_INTERFACE, [vswitch_idl.OVSREC_INTERFACE_COL_NAME])
    schema_helper.register_columns(vswitch_idl.OVSREC_TABLE_QOS, [vswitch_idl.OVSREC_QOS_COL_QUEUES])
    schema_helper.register_columns(vswitch_idl.OVSREC_TABLE_QUEUE, [])