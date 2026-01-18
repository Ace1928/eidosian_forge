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
def test_pinfo_nonascii():
    from . import nonascii2
    ip.user_ns['nonascii2'] = nonascii2
    ip._inspect('pinfo', 'nonascii2', detail_level=1)