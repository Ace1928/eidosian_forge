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
def _cmd_set_fail_mode(self, ctx, command):
    br_name = command.args[0]
    mode = command.args[1]
    if mode not in ('standalone', 'secure'):
        vsctl_fatal('fail-mode must be "standalone" or "secure"')
    self._set_fail_mode(ctx, br_name, mode)