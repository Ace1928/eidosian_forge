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
class ModelCursorWrapper(BaseModelCursorWrapper):

    def __init__(self, cursor, model, select, from_list, joins):
        super(ModelCursorWrapper, self).__init__(cursor, model, select)
        self.from_list = from_list
        self.joins = joins

    def initialize(self):
        self._initialize_columns()
        selected_src = set([field.model for field in self.fields if field is not None])
        select, columns = (self.select, self.columns)
        self.key_to_constructor = {self.model: self.model}
        self.src_is_dest = {}
        self.src_to_dest = []
        accum = collections.deque(self.from_list)
        dests = set()
        while accum:
            curr = accum.popleft()
            if isinstance(curr, Join):
                accum.append(curr.lhs)
                accum.append(curr.rhs)
                continue
            if curr not in self.joins:
                continue
            is_dict = isinstance(curr, dict)
            for key, attr, constructor, join_type in self.joins[curr]:
                if key not in self.key_to_constructor:
                    self.key_to_constructor[key] = constructor
                    self.src_to_dest.append((curr, attr, key, is_dict, join_type))
                    dests.add(key)
                    accum.append(key)
        for src in selected_src:
            if src not in self.key_to_constructor:
                if is_model(src):
                    self.key_to_constructor[src] = src
                elif isinstance(src, ModelAlias):
                    self.key_to_constructor[src] = src.model
        for src, _, dest, _, _ in self.src_to_dest:
            self.src_is_dest[src] = src in dests and (dest in selected_src or src in selected_src)
        self.column_keys = []
        for idx, node in enumerate(select):
            key = self.model
            field = self.fields[idx]
            if field is not None:
                if isinstance(field, FieldAlias):
                    key = field.source
                else:
                    key = field.model
            elif isinstance(node, BindTo):
                if node.dest not in self.key_to_constructor:
                    raise ValueError('%s specifies bind-to %s, but %s is not among the selected sources.' % (node.unwrap(), node.dest, node.dest))
                key = node.dest
            else:
                if isinstance(node, Node):
                    node = node.unwrap()
                if isinstance(node, Column):
                    key = node.source
            self.column_keys.append(key)

    def process_row(self, row):
        objects = {}
        object_list = []
        for key, constructor in self.key_to_constructor.items():
            objects[key] = constructor(__no_default__=True)
            object_list.append(objects[key])
        default_instance = objects[self.model]
        set_keys = set()
        for idx, key in enumerate(self.column_keys):
            instance = objects.get(key, default_instance)
            column = self.columns[idx]
            value = row[idx]
            if value is not None:
                set_keys.add(key)
            if self.converters[idx]:
                value = self.converters[idx](value)
            if isinstance(instance, dict):
                instance[column] = value
            else:
                setattr(instance, column, value)
        for src, attr, dest, is_dict, join_type in self.src_to_dest:
            instance = objects[src]
            try:
                joined_instance = objects[dest]
            except KeyError:
                continue
            if instance is None or dest is None or (dest not in set_keys and (not self.src_is_dest.get(dest))):
                continue
            if instance not in set_keys and dest not in set_keys and join_type.endswith('OUTER JOIN'):
                continue
            if is_dict:
                instance[attr] = joined_instance
            else:
                setattr(instance, attr, joined_instance)
        for instance in object_list:
            if isinstance(instance, Model):
                instance._dirty.clear()
        return objects[self.model]