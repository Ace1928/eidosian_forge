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
class FieldAlias(Field):

    def __init__(self, source, field):
        self.source = source
        self.model = source.model
        self.field = field

    @classmethod
    def create(cls, source, field):

        class _FieldAlias(cls, type(field)):
            pass
        return _FieldAlias(source, field)

    def clone(self):
        return FieldAlias(self.source, self.field)

    def adapt(self, value):
        return self.field.adapt(value)

    def python_value(self, value):
        return self.field.python_value(value)

    def db_value(self, value):
        return self.field.db_value(value)

    def __getattr__(self, attr):
        return self.source if attr == 'model' else getattr(self.field, attr)

    def __sql__(self, ctx):
        return ctx.sql(Column(self.source, self.field.column_name))