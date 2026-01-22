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
def __process_fetch_reply(self, ops):
    update = False
    for fetch_request in self._fetch_requests:
        row = fetch_request['row']
        column_name = fetch_request['column_name']
        index = fetch_request['index']
        table = row._table
        select = ops[index]
        fetched_rows = select.get('rows')
        if not Transaction.__check_json_type(fetched_rows, (list, tuple), '"select" reply "rows"'):
            return False
        if len(fetched_rows) != 1:
            vlog.warn('"select" reply "rows" has %d elements instead of 1' % len(fetched_rows))
            continue
        fetched_row = fetched_rows[0]
        if not Transaction.__check_json_type(fetched_row, (dict,), '"select" reply row'):
            continue
        column = table.columns.get(column_name)
        datum_json = fetched_row.get(column_name)
        datum = data.Datum.from_json(column.type, datum_json)
        row._data[column_name] = datum
        update = True
    return update