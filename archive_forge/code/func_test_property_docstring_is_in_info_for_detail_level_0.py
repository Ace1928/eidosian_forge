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
def test_property_docstring_is_in_info_for_detail_level_0():

    class A(object):

        @property
        def foobar(self):
            """This is `foobar` property."""
            pass
    ip.user_ns['a_obj'] = A()
    assert 'This is `foobar` property.' == ip.object_inspect('a_obj.foobar', detail_level=0)['docstring']
    ip.user_ns['a_cls'] = A
    assert 'This is `foobar` property.' == ip.object_inspect('a_cls.foobar', detail_level=0)['docstring']