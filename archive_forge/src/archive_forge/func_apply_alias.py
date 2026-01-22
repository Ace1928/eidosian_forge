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
def apply_alias(self, ctx):
    if ctx.scope == SCOPE_SOURCE:
        if self._alias:
            ctx.alias_manager[self] = self._alias
        ctx.literal(' AS ').sql(Entity(ctx.alias_manager[self]))
    return ctx