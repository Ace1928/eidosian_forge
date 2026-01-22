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
def _create_table_as(self, table_name, query, safe=True, **meta):
    ctx = self._create_context().literal('CREATE TEMPORARY TABLE ' if meta.get('temporary') else 'CREATE TABLE ')
    if safe:
        ctx.literal('IF NOT EXISTS ')
    return ctx.sql(Entity(*ensure_tuple(table_name))).literal(' AS ').sql(query)