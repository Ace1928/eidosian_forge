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
def _cmd_add_port(self, ctx, command):
    may_exist = command.has_option('--may_exist') or command.has_option('--may-exist')
    br_name = command.args[0]
    port_name = command.args[1]
    iface_names = [command.args[1]]
    settings = [ctx.parse_column_key_value(self.schema.tables[vswitch_idl.OVSREC_TABLE_PORT], setting) for setting in command.args[2:]]
    ctx.add_port(br_name, port_name, may_exist, False, iface_names, settings)