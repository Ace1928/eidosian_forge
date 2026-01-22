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
def _substitute_uuids(self, json):
    if isinstance(json, (list, tuple)):
        if len(json) == 2 and json[0] == 'uuid' and ovs.ovsuuid.is_valid_string(json[1]):
            uuid = ovs.ovsuuid.from_string(json[1])
            row = self._txn_rows.get(uuid, None)
            if row and row._data is None:
                return ['named-uuid', _uuid_name_from_uuid(uuid)]
        else:
            return [self._substitute_uuids(elem) for elem in json]
    return json