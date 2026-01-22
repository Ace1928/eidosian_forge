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
def _wrap_parens(self, ctx, subq):
    csq_setting = ctx.state.compound_select_parentheses
    if not csq_setting or csq_setting == CSQ_PARENTHESES_NEVER:
        return False
    elif csq_setting == CSQ_PARENTHESES_ALWAYS:
        return True
    elif csq_setting == CSQ_PARENTHESES_UNNESTED:
        if ctx.state.in_expr or ctx.state.in_function:
            return False
        return not isinstance(subq, CompoundSelectQuery)