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
class SearchField(Field):

    def __init__(self, unindexed=False, column_name=None, **k):
        if k:
            raise ValueError('SearchField does not accept these keyword arguments: %s.' % sorted(k))
        super(SearchField, self).__init__(unindexed=unindexed, column_name=column_name, null=True)

    def match(self, term):
        return match(self, term)

    @property
    def fts_column_index(self):
        if not hasattr(self, '_fts_column_index'):
            search_fields = [f.name for f in self.model._meta.sorted_fields if isinstance(f, SearchField)]
            self._fts_column_index = search_fields.index(self.name)
        return self._fts_column_index

    def highlight(self, left, right):
        column_idx = self.fts_column_index
        return fn.highlight(self.model._meta.entity, column_idx, left, right)

    def snippet(self, left, right, over_length='...', max_tokens=16):
        if not 0 < max_tokens < 65:
            raise ValueError('max_tokens must be between 1 and 64 (inclusive)')
        column_idx = self.fts_column_index
        return fn.snippet(self.model._meta.entity, column_idx, left, right, over_length, max_tokens)