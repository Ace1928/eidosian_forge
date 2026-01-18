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
def turn_warnings_into_errors():
    warnings_filters_info_list = _get_warnings_filters_info_list()
    for warning_info in warnings_filters_info_list:
        warnings.filterwarnings(warning_info.action, message=warning_info.message, category=warning_info.category)