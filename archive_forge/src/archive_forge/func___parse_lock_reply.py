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
def __parse_lock_reply(self, result):
    self._lock_request_id = None
    got_lock = isinstance(result, dict) and result.get('locked') is True
    self.__update_has_lock(got_lock)
    if not got_lock:
        self.is_lock_contended = True