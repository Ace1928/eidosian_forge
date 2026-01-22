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
def _pre_cmd_set(self, ctx, command):
    table_name = command.args[0]
    table_schema = self.schema.tables[table_name]
    columns = [ctx.parse_column_key_value(table_schema, column_key_value)[0] for column_key_value in command.args[2:]]
    self._pre_mod_columns(ctx, table_name, columns)