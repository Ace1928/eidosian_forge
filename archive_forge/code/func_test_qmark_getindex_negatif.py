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
def test_qmark_getindex_negatif():

    def dummy():
        """
        MARKER 3
        """
    container = [dummy]
    with cleanup_user_ns(container=container):
        with AssertPrints('MARKER 3'):
            ip.run_cell('container[-1]?')
    assert 'container' not in ip.user_ns.keys()