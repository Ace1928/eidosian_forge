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
class MySQLColumn(namedtuple('_Column', ('name', 'definition', 'null', 'pk', 'default', 'extra'))):

    @property
    def is_pk(self):
        return self.pk == 'PRI'

    @property
    def is_unique(self):
        return self.pk == 'UNI'

    @property
    def is_null(self):
        return self.null == 'YES'

    def sql(self, column_name=None, is_null=None):
        if is_null is None:
            is_null = self.is_null
        if column_name is None:
            column_name = self.name
        parts = [Entity(column_name), SQL(self.definition)]
        if self.is_unique:
            parts.append(SQL('UNIQUE'))
        if is_null:
            parts.append(SQL('NULL'))
        else:
            parts.append(SQL('NOT NULL'))
        if self.is_pk:
            parts.append(SQL('PRIMARY KEY'))
        if self.extra:
            parts.append(SQL(self.extra))
        return NodeList(parts)