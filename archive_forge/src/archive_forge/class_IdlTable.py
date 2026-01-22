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
class IdlTable(object):

    def __init__(self, idl, table):
        assert isinstance(table, ovs.db.schema.TableSchema)
        self._table = table
        self.need_table = False
        self.rows = custom_index.IndexedRows(self)
        self.idl = idl
        self._condition_state = ConditionState()
        self.columns = {k: IdlColumn(v) for k, v in table.columns.items()}

    def __getattr__(self, attr):
        return getattr(self._table, attr)

    @property
    def condition_state(self):
        return self._condition_state

    @property
    def condition(self):
        return self.condition_state.latest

    @condition.setter
    def condition(self, condition):
        assert isinstance(condition, list)
        self.idl.cond_change(self.name, condition)

    @classmethod
    def schema_tables(cls, idl, schema):
        return {k: cls(idl, v) for k, v in schema.tables.items()}