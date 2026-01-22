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
class NoopModelSelect(ModelSelect):

    def __sql__(self, ctx):
        return self.model._meta.database.get_noop_select(ctx)

    def _get_cursor_wrapper(self, cursor):
        return CursorWrapper(cursor)