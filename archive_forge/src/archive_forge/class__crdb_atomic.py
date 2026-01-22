import functools
import re
import sys
from peewee import *
from peewee import _atomic
from peewee import _manual
from peewee import ColumnMetadata  # (name, data_type, null, primary_key, table, default)
from peewee import EnclosedNodeList
from peewee import Entity
from peewee import ForeignKeyMetadata  # (column, dest_table, dest_column, table).
from peewee import IndexMetadata
from peewee import NodeList
from playhouse.pool import _PooledPostgresqlDatabase
class _crdb_atomic(_atomic):

    def __enter__(self):
        if self.db.transaction_depth() > 0:
            if not isinstance(self.db.top_transaction(), _manual):
                raise NotImplementedError(TXN_ERR_MSG)
        return super(_crdb_atomic, self).__enter__()