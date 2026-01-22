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
def _cmd_get(self, ctx, command):
    id_ = None
    if_exists = command.has_option('--if-exists')
    table_name = command.args[0]
    record_id = command.args[1]
    column_keys = [ctx.parse_column_key(column_key) for column_key in command.args[2:]]
    command.result = self._get(ctx, table_name, record_id, column_keys, id_, if_exists)