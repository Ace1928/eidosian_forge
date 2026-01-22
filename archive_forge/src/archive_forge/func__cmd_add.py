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
def _cmd_add(self, ctx, command):
    table_name = command.args[0]
    record_id = command.args[1]
    column = command.args[2]
    column_key_value_strings = []
    for value in command.args[3:]:
        if '=' in value:
            column_key_value_strings.append('%s:%s' % (column, value))
        else:
            column_key_value_strings.append('%s=%s' % (column, value))
    table_schema = self.schema.tables[table_name]
    column_values = [ctx.parse_column_key_value(table_schema, column_key_value_string) for column_key_value_string in column_key_value_strings]
    self._add(ctx, table_name, record_id, column_values)