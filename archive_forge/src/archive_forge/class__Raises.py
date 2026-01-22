import atexit
import contextlib
import functools
import importlib
import inspect
import os
import os.path as op
import re
import shutil
import sys
import tempfile
import unittest
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from functools import wraps
from inspect import signature
from subprocess import STDOUT, CalledProcessError, TimeoutExpired, check_output
from unittest import TestCase
import joblib
import numpy as np
import scipy as sp
from numpy.testing import assert_allclose as np_assert_allclose
from numpy.testing import (
import sklearn
from sklearn.utils import (
from sklearn.utils._array_api import _check_array_api_dispatch
from sklearn.utils.fixes import VisibleDeprecationWarning, parse_version, sp_version
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import (
class _Raises(contextlib.AbstractContextManager):

    def __init__(self, expected_exc_type, match, may_pass, err_msg):
        self.expected_exc_types = expected_exc_type if isinstance(expected_exc_type, Iterable) else [expected_exc_type]
        self.matches = [match] if isinstance(match, str) else match
        self.may_pass = may_pass
        self.err_msg = err_msg
        self.raised_and_matched = False

    def __exit__(self, exc_type, exc_value, _):
        if exc_type is None:
            if self.may_pass:
                return True
            else:
                err_msg = self.err_msg or f'Did not raise: {self.expected_exc_types}'
                raise AssertionError(err_msg)
        if not any((issubclass(exc_type, expected_type) for expected_type in self.expected_exc_types)):
            if self.err_msg is not None:
                raise AssertionError(self.err_msg) from exc_value
            else:
                return False
        if self.matches is not None:
            err_msg = self.err_msg or 'The error message should contain one of the following patterns:\n{}\nGot {}'.format('\n'.join(self.matches), str(exc_value))
            if not any((re.search(match, str(exc_value)) for match in self.matches)):
                raise AssertionError(err_msg) from exc_value
            self.raised_and_matched = True
        return True