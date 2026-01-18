import logging
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_utils import excutils
from oslo_utils import timeutils
def test_save_and_reraise_exception_provided_logger(self):
    fake_logger = mock.MagicMock()
    try:
        try:
            raise Exception('foo')
        except Exception:
            with excutils.save_and_reraise_exception(logger=fake_logger):
                raise Exception('second exception')
    except Exception:
        pass
    self.assertTrue(fake_logger.error.called)