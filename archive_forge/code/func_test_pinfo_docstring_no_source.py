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
def test_pinfo_docstring_no_source():
    """Docstring should be included with detail_level=1 if there is no source"""
    with AssertPrints('Docstring:'):
        ip._inspect('pinfo', 'str.format', detail_level=0)
    with AssertPrints('Docstring:'):
        ip._inspect('pinfo', 'str.format', detail_level=1)