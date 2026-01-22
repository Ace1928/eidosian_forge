import collections
import enum
import functools
import uuid
import ovs.db.data as data
import ovs.db.parser
import ovs.db.schema
import ovs.jsonrpc
import ovs.ovsuuid
import ovs.poller
import ovs.vlog
from ovs.db import custom_index
from ovs.db import error
class IdlColumn(object):

    def __init__(self, column):
        self._column = column
        self.alert = True

    def __getattr__(self, attr):
        return getattr(self._column, attr)