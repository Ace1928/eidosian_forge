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
def __check_server_db(self):
    """Returns True if this is a valid server database, False otherwise."""
    session_name = self.session_name()
    if self._server_db_table not in self.server_tables:
        vlog.info('%s: server does not have %s table in its %s database' % (session_name, self._server_db_table, self._server_db_name))
        return False
    rows = self.server_tables[self._server_db_table].rows
    database = None
    for row in rows.values():
        if self.cluster_id:
            if self.cluster_id in map(lambda x: str(x)[:4], row.cid):
                database = row
                break
        elif row.name == self._db.name:
            database = row
            break
    if not database:
        vlog.info('%s: server does not have %s database' % (session_name, self._db.name))
        return False
    if database.model == CLUSTERED:
        if not database.schema:
            vlog.info('%s: clustered database server has not yet joined cluster; trying another server' % session_name)
            return False
        if not database.connected:
            vlog.info('%s: clustered database server is disconnected from cluster; trying another server' % session_name)
            return False
        if self.leader_only and (not database.leader):
            vlog.info('%s: clustered database server is not cluster leader; trying another server' % session_name)
            return False
        if database.index:
            if database.index[0] < self._min_index:
                vlog.warn('%s: clustered database server has stale data; trying another server' % session_name)
                return False
            self._min_index = database.index[0]
    elif database.model == RELAY:
        if not database.schema:
            vlog.info('%s: relay database server has not yet connected to the relay source; trying another server' % session_name)
            return False
        if not database.connected:
            vlog.info('%s: relay database server is disconnected from the relay source; trying another server' % session_name)
            return False
        if self.leader_only:
            vlog.info('%s: relay database server cannot be a leader; trying another server' % session_name)
            return False
    return True