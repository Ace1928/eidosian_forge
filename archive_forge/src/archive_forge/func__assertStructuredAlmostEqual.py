import enum
import glob
import logging
import math
import os
import operator
import re
import subprocess
import sys
from io import StringIO
from unittest import *
import unittest as _unittest
import pytest as pytest
from pyomo.common.collections import Mapping, Sequence
from pyomo.common.dependencies import attempt_import, check_min_version
from pyomo.common.errors import InvalidValueError
from pyomo.common.fileutils import import_file
from pyomo.common.log import LoggingIntercept, pyomo_formatter
from pyomo.common.tee import capture_output
from unittest import mock
def _assertStructuredAlmostEqual(first, second, abstol, reltol, exact, item_callback, exception):
    """Recursive implementation of assertStructuredAlmostEqual"""
    args = (first, second)
    f, s = args
    if all((isinstance(_, Mapping) for _ in args)):
        if exact and len(first) != len(second):
            raise exception('mappings are different sizes (%s != %s)' % (len(first), len(second)))
        for key in first:
            if key not in second:
                raise exception('key (%s) from first not found in second' % (_unittest.case.safe_repr(key),))
            try:
                _assertStructuredAlmostEqual(first[key], second[key], abstol, reltol, exact, item_callback, exception)
            except exception as e:
                raise exception('%s\n    Found when comparing key %s' % (str(e), _unittest.case.safe_repr(key)))
        return
    elif any((isinstance(_, str) for _ in args)):
        if first == second:
            return
    elif all((isinstance(_, Sequence) for _ in args)):
        if exact and len(first) != len(second):
            raise exception('sequences are different sizes (%s != %s)' % (len(first), len(second)))
        for i, (f, s) in enumerate(zip(first, second)):
            try:
                _assertStructuredAlmostEqual(f, s, abstol, reltol, exact, item_callback, exception)
            except exception as e:
                raise exception('%s\n    Found at position %s' % (str(e), i))
        return
    else:
        try:
            if first is second or first == second:
                return
        except:
            pass
        try:
            f = item_callback(first)
            s = item_callback(second)
            if f == s:
                return
            diff = abs(f - s)
            if abstol is not None and diff <= abstol:
                return
            if reltol is not None and diff / max(abs(f), abs(s)) <= reltol:
                return
            if math.isnan(f) and math.isnan(s):
                return
        except:
            pass
    msg = '%s !~= %s' % (_unittest.case.safe_repr(first), _unittest.case.safe_repr(second))
    if f is not first or s is not second:
        msg = '%s !~= %s (%s)' % (_unittest.case.safe_repr(f), _unittest.case.safe_repr(s), msg)
    raise exception(msg)