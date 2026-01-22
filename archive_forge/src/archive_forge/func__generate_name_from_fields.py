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
def _generate_name_from_fields(self, model, fields):
    accum = []
    for field in fields:
        if isinstance(field, basestring):
            accum.append(field.split()[0])
        else:
            if isinstance(field, Node) and (not isinstance(field, Field)):
                field = field.unwrap()
            if isinstance(field, Field):
                accum.append(field.column_name)
    if not accum:
        raise ValueError('Unable to generate a name for the index, please explicitly specify a name.')
    clean_field_names = re.sub('[^\\w]+', '', '_'.join(accum))
    meta = model._meta
    prefix = meta.name if meta.legacy_table_names else meta.table_name
    return _truncate_constraint_name('_'.join((prefix, clean_field_names)))