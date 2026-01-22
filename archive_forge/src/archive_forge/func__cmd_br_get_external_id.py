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
def _cmd_br_get_external_id(self, ctx, command):
    br_name = command.args[0]
    if len(command.args) > 1:
        command.result = self._br_get_external_id_value(ctx, br_name, command.args[1])
    else:
        command.result = self._br_get_external_id_list(ctx, br_name)