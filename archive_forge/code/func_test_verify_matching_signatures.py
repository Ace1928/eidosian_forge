import inspect
import sys
import os
import tempfile
from io import StringIO
from unittest import mock
import numpy as np
from numpy.testing import (
from numpy.core.overrides import (
from numpy.compat import pickle
import pytest
def test_verify_matching_signatures(self):
    verify_matching_signatures(lambda x: 0, lambda x: 0)
    verify_matching_signatures(lambda x=None: 0, lambda x=None: 0)
    verify_matching_signatures(lambda x=1: 0, lambda x=None: 0)
    with assert_raises(RuntimeError):
        verify_matching_signatures(lambda a: 0, lambda b: 0)
    with assert_raises(RuntimeError):
        verify_matching_signatures(lambda x: 0, lambda x=None: 0)
    with assert_raises(RuntimeError):
        verify_matching_signatures(lambda x=None: 0, lambda y=None: 0)
    with assert_raises(RuntimeError):
        verify_matching_signatures(lambda x=1: 0, lambda y=1: 0)