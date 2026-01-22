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
@staticmethod
def _get_table(table_name):
    best_match = None
    best_score = 0
    for table in VSCtl._TABLES:
        score = VSCtl._score_partial_match(table.table_name, table_name)
        if score > best_score:
            best_match = table
            best_score = score
        elif score == best_score:
            best_match = None
    if best_match:
        return best_match
    elif best_score:
        vsctl_fatal('multiple table names match "%s"' % table_name)
    else:
        vsctl_fatal('unknown table "%s"' % table_name)