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
def __do_parse_update(self, table_updates, version, tables):
    if not isinstance(table_updates, dict):
        raise error.Error('<table-updates> is not an object', table_updates)
    notices = []
    for table_name, table_update in table_updates.items():
        table = tables.get(table_name)
        if not table:
            raise error.Error('<table-updates> includes unknown table "%s"' % table_name)
        if not isinstance(table_update, dict):
            raise error.Error('<table-update> for table "%s" is not an object' % table_name, table_update)
        for uuid_string, row_update in table_update.items():
            if not ovs.ovsuuid.is_valid_string(uuid_string):
                raise error.Error('<table-update> for table "%s" contains bad UUID "%s" as member name' % (table_name, uuid_string), table_update)
            uuid = ovs.ovsuuid.from_string(uuid_string)
            if not isinstance(row_update, dict):
                raise error.Error('<table-update> for table "%s" contains <row-update> for %s that is not an object' % (table_name, uuid_string))
            self.cooperative_yield()
            if version in (OVSDB_UPDATE2, OVSDB_UPDATE3):
                changes = self.__process_update2(table, uuid, row_update)
                if changes:
                    notices.append(changes)
                    self.change_seqno += 1
                continue
            parser = ovs.db.parser.Parser(row_update, 'row-update')
            old = parser.get_optional('old', [dict])
            new = parser.get_optional('new', [dict])
            parser.finish()
            if not old and (not new):
                raise error.Error('<row-update> missing "old" and "new" members', row_update)
            changes = self.__process_update(table, uuid, old, new)
            if changes:
                notices.append(changes)
                self.change_seqno += 1
    for notice in notices:
        self.notify(*notice)