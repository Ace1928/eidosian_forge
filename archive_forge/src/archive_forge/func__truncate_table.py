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
def _truncate_table(self, restart_identity=False, cascade=False):
    db = self.database
    if not db.truncate_table:
        return self._create_context().literal('DELETE FROM ').sql(self.model)
    ctx = self._create_context().literal('TRUNCATE TABLE ').sql(self.model)
    if restart_identity:
        ctx = ctx.literal(' RESTART IDENTITY')
    if cascade:
        ctx = ctx.literal(' CASCADE')
    return ctx