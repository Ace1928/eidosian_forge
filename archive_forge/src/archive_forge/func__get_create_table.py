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
def _get_create_table(self, table):
    res = self.database.execute_sql('select name, sql from sqlite_master where type=? and LOWER(name)=?', ['table', table.lower()])
    return res.fetchone()