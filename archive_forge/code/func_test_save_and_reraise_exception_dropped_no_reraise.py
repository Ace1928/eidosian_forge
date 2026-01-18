import logging
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_utils import excutils
from oslo_utils import timeutils
@mock.patch('logging.getLogger')
def test_save_and_reraise_exception_dropped_no_reraise(self, get_logger_mock):
    logger = get_logger_mock()
    e = None
    msg = 'second exception'
    try:
        try:
            raise Exception('dropped')
        except Exception:
            with excutils.save_and_reraise_exception(reraise=False):
                raise Exception(msg)
    except Exception as _e:
        e = _e
    self.assertEqual(str(e), msg)
    self.assertFalse(logger.error.called)