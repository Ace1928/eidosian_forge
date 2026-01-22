from collections import namedtuple
import functools
import hashlib
import re
from peewee import *
from peewee import CommaNodeList
from peewee import EnclosedNodeList
from peewee import Entity
from peewee import Expression
from peewee import Node
from peewee import NodeList
from peewee import OP
from peewee import callable_
from peewee import sort_models
from peewee import sqlite3
from peewee import _truncate_constraint_name
def _primary_key_columns(self, tbl):
    query = "\n            SELECT pg_attribute.attname\n            FROM pg_index, pg_class, pg_attribute\n            WHERE\n                pg_class.oid = '%s'::regclass AND\n                indrelid = pg_class.oid AND\n                pg_attribute.attrelid = pg_class.oid AND\n                pg_attribute.attnum = any(pg_index.indkey) AND\n                indisprimary;\n        "
    cursor = self.database.execute_sql(query % tbl)
    return [row[0] for row in cursor.fetchall()]