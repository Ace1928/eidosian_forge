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
def _get_column(self, table_name, column_name):
    best_match = None
    best_score = 0
    columns = self.schema.tables[table_name].columns.keys()
    for column in columns:
        score = VSCtl._score_partial_match(column, column_name)
        if score > best_score:
            best_match = column
            best_score = score
        elif score == best_score:
            best_match = None
    if best_match:
        return str(best_match)
    elif best_score:
        vsctl_fatal('%s contains more than one column whose name matches "%s"' % (table_name, column_name))
    else:
        vsctl_fatal('%s does not contain a column whose name matches "%s"' % (table_name, column_name))