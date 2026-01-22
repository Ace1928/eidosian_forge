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
def _json_operation(self, func, value, as_json=None):
    if as_json or isinstance(value, (list, dict)):
        value = fn.jsonb(self._field._json_dumps(value))
    return func(self._field, self.path, value)