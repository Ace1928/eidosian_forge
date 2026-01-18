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
def test_pinfo_no_docstring_if_source():
    """Docstring should not be included with detail_level=1 if source is found"""

    def foo():
        """foo has a docstring"""
    ip.user_ns['foo'] = foo
    with AssertPrints('Docstring:'):
        ip._inspect('pinfo', 'foo', detail_level=0)
    with AssertPrints('Source:'):
        ip._inspect('pinfo', 'foo', detail_level=1)
    with AssertNotPrints('Docstring:'):
        ip._inspect('pinfo', 'foo', detail_level=1)