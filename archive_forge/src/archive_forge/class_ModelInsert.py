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
class ModelInsert(_ModelWriteQueryHelper, Insert):
    default_row_type = ROW.TUPLE

    def __init__(self, *args, **kwargs):
        super(ModelInsert, self).__init__(*args, **kwargs)
        if self._returning is None and self.model._meta.database is not None:
            if self.model._meta.database.returning_clause:
                self._returning = self.model._meta.get_primary_keys()

    def returning(self, *returning):
        if returning and self._row_type is None:
            self._row_type = ROW.MODEL
        return super(ModelInsert, self).returning(*returning)

    def get_default_data(self):
        return self.model._meta.defaults

    def get_default_columns(self):
        fields = self.model._meta.sorted_fields
        return fields[1:] if self.model._meta.auto_increment else fields