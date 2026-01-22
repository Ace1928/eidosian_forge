import collections
import errno
import uuid
from ovs import jsonrpc
from ovs import poller
from ovs import reconnect
from ovs import stream
from ovs import timeval
from ovs.db import idl
from os_ken.base import app_manager
from os_ken.lib import hub
from os_ken.services.protocols.ovsdb import event
from os_ken.services.protocols.ovsdb import model
class Idl(idl.Idl):

    def __init__(self, session, schema):
        if not isinstance(schema, idl.SchemaHelper):
            schema = idl.SchemaHelper(schema_json=schema)
            schema.register_all()
        schema = schema.get_idl_schema()
        self._events = []
        self.tables = schema.tables
        self.readonly = schema.readonly
        self._db = schema
        self._session = session
        self._monitor_request_id = None
        self._last_seqno = None
        self.change_seqno = 0
        self.uuid = uuid.uuid1()
        self.state = self.IDL_S_INITIAL
        self.lock_name = None
        self.has_lock = False
        self.is_lock_contended = False
        self._lock_request_id = None
        self.txn = None
        self._outstanding_txns = {}
        for table in schema.tables.values():
            for column in table.columns.values():
                if not hasattr(column, 'alert'):
                    column.alert = True
            table.need_table = False
            table.rows = {}
            table.idl = self
            table.condition = []
            table.cond_changed = False

    @property
    def events(self):
        events = self._events
        self._events = []
        return events

    def __process_update(self, table, uuid, old, new):
        old_row = table.rows.get(uuid)
        if old_row is not None:
            old_row = model.Row(dictify(old_row))
            old_row['_uuid'] = uuid
        changed = idl.Idl.__process_update(self, table, uuid, old, new)
        if changed:
            if not new:
                ev = (event.EventRowDelete, (table.name, old_row))
            elif not old:
                new_row = model.Row(dictify(table.rows.get(uuid)))
                new_row['_uuid'] = uuid
                ev = (event.EventRowInsert, (table.name, new_row))
            else:
                new_row = model.Row(dictify(table.rows.get(uuid)))
                new_row['_uuid'] = uuid
                ev = (event.EventRowUpdate, (table.name, old_row, new_row))
            self._events.append(ev)
        return changed