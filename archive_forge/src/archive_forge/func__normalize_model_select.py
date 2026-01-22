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
def _normalize_model_select(fields_or_models):
    fields = []
    for fm in fields_or_models:
        if is_model(fm):
            fields.extend(fm._meta.sorted_fields)
        elif isinstance(fm, ModelAlias):
            fields.extend(fm.get_field_aliases())
        elif isinstance(fm, Table) and fm._columns:
            fields.extend([getattr(fm, col) for col in fm._columns])
        else:
            fields.append(fm)
    return fields