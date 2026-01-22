import io
import logging
import time
import testtools
from testtools import TestCase
from fixtures import (
class FakeLoggerTest(TestCase, TestWithFixtures):

    def setUp(self):
        super(FakeLoggerTest, self).setUp()
        self.logger = logging.getLogger()
        self.addCleanup(self.removeHandlers, self.logger)

    def removeHandlers(self, logger):
        for handler in logger.handlers:
            logger.removeHandler(handler)

    def test_output_property_has_output(self):
        fixture = self.useFixture(FakeLogger())
        logging.info('some message')
        self.assertEqual('some message\n', fixture.output)

    def test_replace_and_restore_handlers(self):
        stream = io.StringIO()
        logger = logging.getLogger()
        logger.addHandler(logging.StreamHandler(stream))
        logger.setLevel(logging.INFO)
        logging.info('one')
        fixture = FakeLogger()
        with fixture:
            logging.info('two')
        logging.info('three')
        self.assertEqual('two\n', fixture.output)
        self.assertEqual('one\nthree\n', stream.getvalue())

    def test_preserving_existing_handlers(self):
        stream = io.StringIO()
        self.logger.addHandler(logging.StreamHandler(stream))
        self.logger.setLevel(logging.INFO)
        fixture = FakeLogger(nuke_handlers=False)
        with fixture:
            logging.info('message')
        self.assertEqual('message\n', fixture.output)
        self.assertEqual('message\n', stream.getvalue())

    def test_logging_level_restored(self):
        self.logger.setLevel(logging.DEBUG)
        fixture = FakeLogger(level=logging.WARNING)
        with fixture:
            logging.debug('debug message')
            self.assertEqual(logging.WARNING, self.logger.level)
        self.assertEqual('', fixture.output)
        self.assertEqual(logging.DEBUG, self.logger.level)

    def test_custom_format(self):
        fixture = FakeLogger(format='%(module)s')
        self.useFixture(fixture)
        logging.info('message')
        self.assertEqual('test_logger\n', fixture.output)

    def test_custom_datefmt(self):
        fixture = FakeLogger(format='%(asctime)s %(module)s', datefmt='%Y')
        self.useFixture(fixture)
        logging.info('message')
        self.assertEqual(time.strftime('%Y test_logger\n', time.localtime()), fixture.output)

    def test_custom_formatter(self):
        fixture = FakeLogger(format='%(asctime)s %(module)s', formatter=FooFormatter, datefmt='%Y')
        self.useFixture(fixture)
        logging.info('message')
        self.assertEqual(time.strftime('Foo %Y test_logger\n', time.localtime()), fixture.output)

    def test_logging_output_included_in_details(self):
        fixture = FakeLogger()
        detail_name = "pythonlogging:''"
        with fixture:
            content = fixture.getDetails()[detail_name]
            logging.info('some message')
            self.assertEqual('some message\n', content.as_text())
        self.assertEqual('some message\n', content.as_text())
        with fixture:
            self.assertEqual('', fixture.getDetails()[detail_name].as_text())
        try:
            self.assertEqual('some message\n', content.as_text())
        except AssertionError:
            raise
        except:
            pass

    def test_exceptionraised(self):
        with FakeLogger():
            with testtools.ExpectedException(TypeError):
                logging.info('Some message', 'wrongarg')

    def test_output_can_be_reset(self):
        fixture = FakeLogger()
        with fixture:
            logging.info('message')
        fixture.reset_output()
        self.assertEqual('', fixture.output)