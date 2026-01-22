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
class DeferredForeignKey(Field):
    _unresolved = set()

    def __init__(self, rel_model_name, **kwargs):
        self.field_kwargs = kwargs
        self.rel_model_name = rel_model_name.lower()
        DeferredForeignKey._unresolved.add(self)
        super(DeferredForeignKey, self).__init__(column_name=kwargs.get('column_name'), null=kwargs.get('null'), primary_key=kwargs.get('primary_key'))
    __hash__ = object.__hash__

    def __deepcopy__(self, memo=None):
        return DeferredForeignKey(self.rel_model_name, **self.field_kwargs)

    def set_model(self, rel_model):
        field = ForeignKeyField(rel_model, _deferred=True, **self.field_kwargs)
        if field.primary_key:
            self.model._meta.set_primary_key(self.name, field)
        else:
            self.model._meta.add_field(self.name, field)

    @staticmethod
    def resolve(model_cls):
        unresolved = sorted(DeferredForeignKey._unresolved, key=operator.attrgetter('_order'))
        for dr in unresolved:
            if dr.rel_model_name == model_cls.__name__.lower():
                dr.set_model(model_cls)
                DeferredForeignKey._unresolved.discard(dr)