import logging
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_utils import excutils
from oslo_utils import timeutils
def test_save_and_reraise_exception_capture_reraise(self):

    def _force_reraise():
        try:
            raise IOError('I broke')
        except Exception:
            excutils.save_and_reraise_exception().capture().force_reraise()
    self.assertRaises(IOError, _force_reraise)