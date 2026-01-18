import logging
from oslo_log import fixture
from oslotest import base as test_base
def test_set_before(self):
    logger = logging.getLogger('no-such-logger-set')
    logger.setLevel(logging.ERROR)
    self.assertEqual(logging.ERROR, logger.level)
    fix = fixture.SetLogLevel(['no-such-logger-set'], logging.DEBUG)
    with fix:
        self.assertEqual(logging.DEBUG, logger.level)
    self.assertEqual(logging.ERROR, logger.level)