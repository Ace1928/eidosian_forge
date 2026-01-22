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
def __send_monitor_request(self, max_version=Monitor.monitor_cond_since):
    if self.state == self.IDL_S_INITIAL:
        self.state = self.IDL_S_DATA_MONITOR_COND_REQUESTED
        method = 'monitor_cond'
    elif self.state == self.IDL_S_SERVER_MONITOR_REQUESTED:
        self.state = self.monitor_map[Monitor(max_version)]
        method = Monitor(max_version).name
    else:
        self.state = self.IDL_S_DATA_MONITOR_REQUESTED
        method = 'monitor'
    monitor_requests = {}
    for table in self.tables.values():
        columns = []
        for column in table.columns.keys():
            if table.name not in self.readonly or (table.name in self.readonly and column not in self.readonly[table.name]):
                columns.append(column)
        monitor_request = {'columns': columns}
        if method in ('monitor_cond', 'monitor_cond_since') and (not ConditionState.is_true(table.condition_state.acked)):
            monitor_request['where'] = table.condition_state.acked
        monitor_requests[table.name] = [monitor_request]
    args = [self._db.name, str(self.uuid), monitor_requests]
    if method == 'monitor_cond_since':
        args.append(str(self.last_id))
    msg = ovs.jsonrpc.Message.create_request(method, args)
    self._monitor_request_id = msg.id
    self.send_request(msg)