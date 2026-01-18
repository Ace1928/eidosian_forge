import logging
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_utils import excutils
from oslo_utils import timeutils
def test_raise_other_func_call(self):

    @excutils.exception_filter
    def translate_exceptions(ex):
        raise RuntimeError
    try:
        assert False, 'This is a test'
    except Exception as exc:
        self.assertRaises(RuntimeError, translate_exceptions, exc)