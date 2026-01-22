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
def _create_foreign_key(self, field):
    name = 'fk_%s_%s_refs_%s' % (field.model._meta.table_name, field.column_name, field.rel_model._meta.table_name)
    return self._create_context().literal('ALTER TABLE ').sql(field.model).literal(' ADD CONSTRAINT ').sql(Entity(_truncate_constraint_name(name))).literal(' ').sql(field.foreign_key_constraint())