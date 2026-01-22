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
class SqliteExtDatabase(SqliteDatabase):

    def __init__(self, database, c_extensions=None, rank_functions=True, hash_functions=False, regexp_function=False, bloomfilter=False, json_contains=False, *args, **kwargs):
        super(SqliteExtDatabase, self).__init__(database, *args, **kwargs)
        self._row_factory = None
        if c_extensions and (not CYTHON_SQLITE_EXTENSIONS):
            raise ImproperlyConfigured('SqliteExtDatabase initialized with C extensions, but shared library was not found!')
        prefer_c = CYTHON_SQLITE_EXTENSIONS and c_extensions is not False
        if rank_functions:
            if prefer_c:
                register_rank_functions(self)
            else:
                self.register_function(bm25, 'fts_bm25')
                self.register_function(rank, 'fts_rank')
                self.register_function(bm25, 'fts_bm25f')
                self.register_function(bm25, 'fts_lucene')
        if hash_functions:
            if not prefer_c:
                raise ValueError('C extension required to register hash functions.')
            register_hash_functions(self)
        if regexp_function:
            self.register_function(_sqlite_regexp, 'regexp', 2)
        if bloomfilter:
            if not prefer_c:
                raise ValueError('C extension required to use bloomfilter.')
            register_bloomfilter(self)
        if json_contains:
            self.register_function(_json_contains, 'json_contains')
        self._c_extensions = prefer_c

    def _add_conn_hooks(self, conn):
        super(SqliteExtDatabase, self)._add_conn_hooks(conn)
        if self._row_factory:
            conn.row_factory = self._row_factory

    def row_factory(self, fn):
        self._row_factory = fn