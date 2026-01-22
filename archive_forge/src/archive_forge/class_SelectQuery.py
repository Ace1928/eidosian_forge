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
class SelectQuery(Query):
    union_all = __add__ = __compound_select__('UNION ALL')
    union = __or__ = __compound_select__('UNION')
    intersect = __and__ = __compound_select__('INTERSECT')
    except_ = __sub__ = __compound_select__('EXCEPT')
    __radd__ = __compound_select__('UNION ALL', inverted=True)
    __ror__ = __compound_select__('UNION', inverted=True)
    __rand__ = __compound_select__('INTERSECT', inverted=True)
    __rsub__ = __compound_select__('EXCEPT', inverted=True)

    def select_from(self, *columns):
        if not columns:
            raise ValueError('select_from() must specify one or more columns.')
        query = Select((self,), columns).bind(self._database)
        if getattr(self, 'model', None) is not None:
            query = query.objects(self.model)
        return query