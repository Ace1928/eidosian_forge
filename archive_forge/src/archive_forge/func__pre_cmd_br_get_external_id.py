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
def _pre_cmd_br_get_external_id(self, ctx, _command):
    table_name = vswitch_idl.OVSREC_TABLE_BRIDGE
    columns = [vswitch_idl.OVSREC_BRIDGE_COL_EXTERNAL_IDS]
    self._pre_get_columns(ctx, table_name, columns)