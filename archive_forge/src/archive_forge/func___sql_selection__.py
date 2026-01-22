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
def __sql_selection__(self, ctx, is_subquery=False):
    if self._is_default and is_subquery and (len(self._returning) > 1) and (self.model._meta.primary_key is not False):
        return ctx.sql(self.model._meta.primary_key)
    return ctx.sql(CommaNodeList(self._returning))