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
class CompositeKey(MetaField):
    sequence = None

    def __init__(self, *field_names):
        self.field_names = field_names
        self._safe_field_names = None

    @property
    def safe_field_names(self):
        if self._safe_field_names is None:
            if self.model is None:
                return self.field_names
            self._safe_field_names = [self.model._meta.fields[f].safe_name for f in self.field_names]
        return self._safe_field_names

    def __get__(self, instance, instance_type=None):
        if instance is not None:
            return tuple([getattr(instance, f) for f in self.safe_field_names])
        return self

    def __set__(self, instance, value):
        if not isinstance(value, (list, tuple)):
            raise TypeError('A list or tuple must be used to set the value of a composite primary key.')
        if len(value) != len(self.field_names):
            raise ValueError('The length of the value must equal the number of columns of the composite primary key.')
        for idx, field_value in enumerate(value):
            setattr(instance, self.field_names[idx], field_value)

    def __eq__(self, other):
        expressions = [self.model._meta.fields[field] == value for field, value in zip(self.field_names, other)]
        return reduce(operator.and_, expressions)

    def __ne__(self, other):
        return ~(self == other)

    def __hash__(self):
        return hash((self.model.__name__, self.field_names))

    def __sql__(self, ctx):
        parens = ctx.scope != SCOPE_SOURCE
        return ctx.sql(NodeList([self.model._meta.fields[field] for field in self.field_names], ', ', parens))

    def bind(self, model, name, set_attribute=True):
        self.model = model
        self.column_name = self.name = self.safe_name = name
        setattr(model, self.name, self)