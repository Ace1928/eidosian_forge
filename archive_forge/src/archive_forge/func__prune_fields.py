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
def _prune_fields(self, field_dict, only):
    new_data = {}
    for field in only:
        if isinstance(field, basestring):
            field = self._meta.combined[field]
        if field.name in field_dict:
            new_data[field.name] = field_dict[field.name]
    return new_data