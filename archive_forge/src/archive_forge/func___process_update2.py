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
def __process_update2(self, table, uuid, row_update):
    """Returns Notice if a column changed, False otherwise."""
    row = table.rows.get(uuid)
    if 'delete' in row_update:
        if row:
            del table.rows[uuid]
            return Notice(ROW_DELETE, row)
        else:
            vlog.warn('cannot delete missing row %s from table%s' % (uuid, table.name))
    elif 'insert' in row_update or 'initial' in row_update:
        if row:
            vlog.warn('cannot add existing row %s from table %s' % (uuid, table.name))
            del table.rows[uuid]
        row = self.__create_row(table, uuid)
        if 'insert' in row_update:
            row_update = row_update['insert']
        else:
            row_update = row_update['initial']
        self.__add_default(table, row_update)
        changed = self.__row_update(table, row, row_update)
        table.rows[uuid] = row
        if changed:
            return Notice(ROW_CREATE, row)
    elif 'modify' in row_update:
        if not row:
            raise error.Error('Modify non-existing row')
        old_row = self.__apply_diff(table, row, row_update['modify'])
        return Notice(ROW_UPDATE, row, Row(self, table, uuid, old_row))
    else:
        raise error.Error('<row-update> unknown operation', row_update)
    return False