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
def __row_update(self, table, row, row_json):
    changed = False
    for column_name, datum_json in row_json.items():
        column = table.columns.get(column_name)
        if not column:
            vlog.warn('unknown column %s updating table %s' % (column_name, table.name))
            continue
        try:
            datum = data.Datum.from_json(column.type, datum_json)
        except error.Error as e:
            vlog.warn('error parsing column %s in table %s: %s' % (column_name, table.name, e))
            continue
        if datum != row._data[column_name]:
            row._data[column_name] = datum
            if column.alert:
                changed = True
        else:
            pass
    return changed