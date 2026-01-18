from contextlib import contextmanager
from inspect import signature, Signature, Parameter
import inspect
import os
import pytest
import re
import sys
from .. import oinspect
from decorator import decorator
from IPython.testing.tools import AssertPrints, AssertNotPrints
from IPython.utils.path import compress_user
def test_pinfo_type():
    """
    type can fail in various edge case, for example `type.__subclass__()`
    """
    ip._inspect('pinfo', 'type')