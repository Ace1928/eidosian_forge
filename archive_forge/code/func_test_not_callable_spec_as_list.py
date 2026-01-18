import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_not_callable_spec_as_list(self):
    spec = ('foo', 'bar')
    p = patch(MODNAME, spec=spec)
    m = p.start()
    try:
        self.assertFalse(callable(m))
    finally:
        p.stop()