import io
import logging
import time
import testtools
from testtools import TestCase
from fixtures import (
class LogHandlerTest(TestCase, TestWithFixtures):

    class CustomHandler(logging.Handler):

        def __init__(self, *args, **kwargs):
            """Create the instance, and add a records attribute."""
            logging.Handler.__init__(self, *args, **kwargs)
            self.msgs = []

        def emit(self, record):
            self.msgs.append(record.msg)

    def setUp(self):
        super(LogHandlerTest, self).setUp()
        self.logger = logging.getLogger()
        self.addCleanup(self.removeHandlers, self.logger)

    def removeHandlers(self, logger):
        for handler in logger.handlers:
            logger.removeHandler(handler)

    def test_captures_logging(self):
        fixture = self.useFixture(LogHandler(self.CustomHandler()))
        logging.info('some message')
        self.assertEqual(['some message'], fixture.handler.msgs)

    def test_replace_and_restore_handlers(self):
        stream = io.StringIO()
        logger = logging.getLogger()
        logger.addHandler(logging.StreamHandler(stream))
        logger.setLevel(logging.INFO)
        logging.info('one')
        fixture = LogHandler(self.CustomHandler())
        with fixture:
            logging.info('two')
        logging.info('three')
        self.assertEqual(['two'], fixture.handler.msgs)
        self.assertEqual('one\nthree\n', stream.getvalue())

    def test_preserving_existing_handlers(self):
        stream = io.StringIO()
        self.logger.addHandler(logging.StreamHandler(stream))
        self.logger.setLevel(logging.INFO)
        fixture = LogHandler(self.CustomHandler(), nuke_handlers=False)
        with fixture:
            logging.info('message')
        self.assertEqual(['message'], fixture.handler.msgs)
        self.assertEqual('message\n', stream.getvalue())

    def test_logging_level_restored(self):
        self.logger.setLevel(logging.DEBUG)
        fixture = LogHandler(self.CustomHandler(), level=logging.WARNING)
        with fixture:
            logging.debug('debug message')
            self.assertEqual(logging.WARNING, self.logger.level)
        self.assertEqual([], fixture.handler.msgs)
        self.assertEqual(logging.DEBUG, self.logger.level)