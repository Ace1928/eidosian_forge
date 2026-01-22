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
def _normalize_join(self, src, dest, on, attr):
    on_alias = isinstance(on, Alias)
    if on_alias:
        attr = attr or on._alias
        on = on.alias()
    src_model, src_is_model = self._get_model(src)
    dest_model, dest_is_model = self._get_model(dest)
    if src_model and dest_model:
        self._join_ctx = dest
        constructor = dest_model
        if not (src_is_model and dest_is_model) and isinstance(on, Column):
            if on.source is src:
                to_field = src_model._meta.columns[on.name]
            elif on.source is dest:
                to_field = dest_model._meta.columns[on.name]
            else:
                raise AttributeError('"on" clause Column %s does not belong to %s or %s.' % (on, src_model, dest_model))
            on = None
        elif isinstance(on, Field):
            to_field = on
            on = None
        else:
            to_field = None
        fk_field, is_backref = self._generate_on_clause(src_model, dest_model, to_field, on)
        if on is None:
            src_attr = 'name' if src_is_model else 'column_name'
            dest_attr = 'name' if dest_is_model else 'column_name'
            if is_backref:
                lhs = getattr(dest, getattr(fk_field, dest_attr))
                rhs = getattr(src, getattr(fk_field.rel_field, src_attr))
            else:
                lhs = getattr(src, getattr(fk_field, src_attr))
                rhs = getattr(dest, getattr(fk_field.rel_field, dest_attr))
            on = lhs == rhs
        if not attr:
            if fk_field is not None and (not is_backref):
                attr = fk_field.name
            else:
                attr = dest_model._meta.name
        elif on_alias and fk_field is not None and (attr == fk_field.object_id_name) and (not is_backref):
            raise ValueError('Cannot assign join alias to "%s", as this attribute is the object_id_name for the foreign-key field "%s"' % (attr, fk_field))
    elif isinstance(dest, Source):
        constructor = dict
        attr = attr or dest._alias
        if not attr and isinstance(dest, Table):
            attr = attr or dest.__name__
    return (on, attr, constructor)