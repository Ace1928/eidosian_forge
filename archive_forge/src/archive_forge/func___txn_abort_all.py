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
def __txn_abort_all(self):
    while self._outstanding_txns:
        txn = self._outstanding_txns.popitem()[1]
        txn._status = Transaction.TRY_AGAIN