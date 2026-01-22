import json
import math
import re
import struct
import sys
from peewee import *
from peewee import ColumnBase
from peewee import EnclosedNodeList
from peewee import Entity
from peewee import Expression
from peewee import Insert
from peewee import Node
from peewee import NodeList
from peewee import OP
from peewee import VirtualField
from peewee import merge_dict
from peewee import sqlite3
def __dbstatus__(flag, return_highwater=False, return_current=False):
    """
        Expose a sqlite3_dbstatus() call for a particular flag as a property of
        the Database instance. Unlike sqlite3_status(), the dbstatus properties
        pertain to the current connection.
        """

    def getter(self):
        if self._state.conn is None:
            raise ImproperlyConfigured('database connection not opened.')
        result = sqlite_get_db_status(self._state.conn, flag)
        if return_current:
            return result[0]
        return result[1] if return_highwater else result
    return property(getter)