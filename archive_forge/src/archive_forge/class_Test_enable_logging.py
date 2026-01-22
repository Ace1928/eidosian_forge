import concurrent.futures
import hashlib
import logging
import sys
from unittest import mock
import fixtures
import os_service_types
import testtools
import openstack
from openstack import exceptions
from openstack.tests.unit import base
from openstack import utils
class Test_enable_logging(base.TestCase):

    def setUp(self):
        super(Test_enable_logging, self).setUp()
        self.openstack_logger = mock.Mock()
        self.openstack_logger.handlers = []
        self.ksa_logger_root = mock.Mock()
        self.ksa_logger_root.handlers = []
        self.ksa_logger_1 = mock.Mock()
        self.ksa_logger_1.handlers = []
        self.ksa_logger_2 = mock.Mock()
        self.ksa_logger_2.handlers = []
        self.ksa_logger_3 = mock.Mock()
        self.ksa_logger_3.handlers = []
        self.urllib3_logger = mock.Mock()
        self.urllib3_logger.handlers = []
        self.stevedore_logger = mock.Mock()
        self.stevedore_logger.handlers = []
        self.fake_get_logger = mock.Mock()
        self.fake_get_logger.side_effect = [self.openstack_logger, self.ksa_logger_root, self.urllib3_logger, self.stevedore_logger, self.ksa_logger_1, self.ksa_logger_2, self.ksa_logger_3]
        self.useFixture(fixtures.MonkeyPatch('logging.getLogger', self.fake_get_logger))

    def _console_tests(self, level, debug, stream):
        openstack.enable_logging(debug=debug, stream=stream)
        self.assertEqual(self.openstack_logger.addHandler.call_count, 1)
        self.openstack_logger.setLevel.assert_called_with(level)

    def _file_tests(self, level, debug):
        file_handler = mock.Mock()
        self.useFixture(fixtures.MonkeyPatch('logging.FileHandler', file_handler))
        fake_path = 'fake/path.log'
        openstack.enable_logging(debug=debug, path=fake_path)
        file_handler.assert_called_with(fake_path)
        self.assertEqual(self.openstack_logger.addHandler.call_count, 1)
        self.openstack_logger.setLevel.assert_called_with(level)

    def test_none(self):
        openstack.enable_logging(debug=True)
        self.fake_get_logger.assert_has_calls([])
        self.openstack_logger.setLevel.assert_called_with(logging.DEBUG)
        self.assertEqual(self.openstack_logger.addHandler.call_count, 1)
        self.assertIsInstance(self.openstack_logger.addHandler.call_args_list[0][0][0], logging.StreamHandler)

    def test_debug_console_stderr(self):
        self._console_tests(logging.DEBUG, True, sys.stderr)

    def test_warning_console_stderr(self):
        self._console_tests(logging.INFO, False, sys.stderr)

    def test_debug_console_stdout(self):
        self._console_tests(logging.DEBUG, True, sys.stdout)

    def test_warning_console_stdout(self):
        self._console_tests(logging.INFO, False, sys.stdout)

    def test_debug_file(self):
        self._file_tests(logging.DEBUG, True)

    def test_warning_file(self):
        self._file_tests(logging.INFO, False)