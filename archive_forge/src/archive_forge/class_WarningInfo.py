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
@dataclass
class WarningInfo:
    action: 'warnings._ActionKind'
    message: str = ''
    category: type[Warning] = Warning

    def to_filterwarning_str(self):
        if self.category.__module__ == 'builtins':
            category = self.category.__name__
        else:
            category = f'{self.category.__module__}.{self.category.__name__}'
        return f'{self.action}:{self.message}:{category}'