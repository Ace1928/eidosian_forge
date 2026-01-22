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
def __update_has_lock(self, new_has_lock):
    if new_has_lock and (not self.has_lock):
        if self._monitor_request_id is None:
            self.change_seqno += 1
        else:
            pass
        self.is_lock_contended = False
    self.has_lock = new_has_lock