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
@staticmethod
def _apply_lookup_to_query(query, key, lookup):
    if isinstance(lookup, slice):
        expr = LSMTable.slice_to_expr(key, lookup)
        if expr is not None:
            query = query.where(expr)
        return (query, False)
    elif isinstance(lookup, Expression):
        return (query.where(lookup), False)
    else:
        return (query.where(key == lookup), True)