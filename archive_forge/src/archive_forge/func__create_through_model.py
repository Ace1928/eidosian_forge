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
def _create_through_model(self):
    lhs, rhs = self.get_models()
    tables = [model._meta.table_name for model in (lhs, rhs)]

    class Meta:
        database = self.model._meta.database
        schema = self.model._meta.schema
        table_name = '%s_%s_through' % tuple(tables)
        indexes = (((lhs._meta.name, rhs._meta.name), True),)
    params = {'on_delete': self._on_delete, 'on_update': self._on_update}
    attrs = {lhs._meta.name: ForeignKeyField(lhs, **params), rhs._meta.name: ForeignKeyField(rhs, **params), 'Meta': Meta}
    klass_name = '%s%sThrough' % (lhs.__name__, rhs.__name__)
    return type(klass_name, (Model,), attrs)