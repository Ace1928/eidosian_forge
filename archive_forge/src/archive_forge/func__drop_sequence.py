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
def _drop_sequence(self, field):
    self._check_sequences(field)
    if self.database.sequence_exists(field.sequence):
        return self._create_context().literal('DROP SEQUENCE ').sql(self._sequence_for_field(field))