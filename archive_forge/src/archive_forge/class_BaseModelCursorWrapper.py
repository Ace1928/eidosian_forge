from bisect import bisect_left
from bisect import bisect_right
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from inspect import isclass
import calendar
import collections
import datetime
import decimal
import hashlib
import itertools
import logging
import operator
import re
import socket
import struct
import sys
import threading
import time
import uuid
import warnings
class BaseModelCursorWrapper(DictCursorWrapper):

    def __init__(self, cursor, model, columns):
        super(BaseModelCursorWrapper, self).__init__(cursor)
        self.model = model
        self.select = columns or []

    def _initialize_columns(self):
        combined = self.model._meta.combined
        table = self.model._meta.table
        description = self.cursor.description
        self.ncols = len(self.cursor.description)
        self.columns = []
        self.converters = converters = [None] * self.ncols
        self.fields = fields = [None] * self.ncols
        for idx, description_item in enumerate(description):
            column = orig_column = description_item[0]
            dot_index = column.rfind('.')
            if dot_index != -1:
                column = column[dot_index + 1:]
            column = column.strip('()"`')
            self.columns.append(column)
            try:
                raw_node = self.select[idx]
            except IndexError:
                if column in combined:
                    raw_node = node = combined[column]
                else:
                    continue
            else:
                node = raw_node.unwrap()
            is_alias = raw_node.is_alias()
            if is_alias:
                self.columns[idx] = orig_column
            if isinstance(node, Field):
                if raw_node._coerce:
                    converters[idx] = node.python_value
                fields[idx] = node
                if not is_alias:
                    self.columns[idx] = node.name
            elif isinstance(node, ColumnBase) and raw_node._converter:
                converters[idx] = raw_node._converter
            elif isinstance(node, Function) and node._coerce:
                if node._python_value is not None:
                    converters[idx] = node._python_value
                elif node.arguments and isinstance(node.arguments[0], Node):
                    first = node.arguments[0].unwrap()
                    if isinstance(first, Entity):
                        path = first._path[-1]
                        first = combined.get(path)
                    if isinstance(first, Field):
                        converters[idx] = safe_python_value(first.python_value)
            elif column in combined:
                if node._coerce:
                    converters[idx] = combined[column].python_value
                if isinstance(node, Column) and node.source == table:
                    fields[idx] = combined[column]
    initialize = _initialize_columns

    def process_row(self, row):
        raise NotImplementedError