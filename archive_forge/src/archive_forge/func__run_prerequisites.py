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
def _run_prerequisites(self, commands):
    schema_helper = self.schema_helper
    schema_helper.register_table(vswitch_idl.OVSREC_TABLE_OPEN_VSWITCH)
    if self.wait_for_reload:
        schema_helper.register_columns(vswitch_idl.OVSREC_TABLE_OPEN_VSWITCH, [vswitch_idl.OVSREC_OPEN_VSWITCH_COL_CUR_CFG])
    for command in commands:
        if not command._prerequisite:
            continue
        ctx = VSCtlContext(None, None, None)
        command._prerequisite(ctx, command)
        ctx.done()