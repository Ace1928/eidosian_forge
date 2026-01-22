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
def _create_table_option_sql(self, options):
    accum = []
    options = merge_dict(self.model._meta.options or {}, options)
    if not options:
        return accum
    for key, value in sorted(options.items()):
        if not isinstance(value, Node):
            if is_model(value):
                value = value._meta.table
            else:
                value = SQL(str(value))
        accum.append(NodeList((SQL(key), value), glue='='))
    return accum