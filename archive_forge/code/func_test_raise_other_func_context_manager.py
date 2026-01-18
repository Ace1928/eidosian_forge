import logging
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_utils import excutils
from oslo_utils import timeutils
def test_raise_other_func_context_manager(self):

    @excutils.exception_filter
    def translate_exceptions(ex):
        raise RuntimeError

    def try_assertion():
        with translate_exceptions:
            assert False, 'This is a test'
    self.assertRaises(RuntimeError, try_assertion)