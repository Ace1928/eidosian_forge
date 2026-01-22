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
def _cmd_list_ifaces_verbose(self, ctx, command):
    datapath_id = command.args[0]
    port_name = None
    if len(command.args) >= 2:
        port_name = command.args[1]
    LOG.debug('command.args %s', command.args)
    iface_cfgs = self._list_ifaces_verbose(ctx, datapath_id, port_name)
    command.result = sorted(iface_cfgs)