import os
import tempfile
import unittest
from webtest import TestApp
import pecan
from pecan.tests import PecanTestCase
class TestConfFromEnv(PecanTestCase):

    def setUp(self):
        super(TestConfFromEnv, self).setUp()
        self.addCleanup(self._remove_config_key)
        from pecan import configuration
        self.get_conf_path_from_env = configuration.get_conf_path_from_env

    def _remove_config_key(self):
        os.environ.pop('PECAN_CONFIG', None)

    def test_invalid_path(self):
        os.environ['PECAN_CONFIG'] = '/'
        msg = 'PECAN_CONFIG was set to an invalid path: /'
        self.assertRaisesRegex(RuntimeError, msg, self.get_conf_path_from_env)

    def test_is_not_set(self):
        msg = 'PECAN_CONFIG is not set and no config file was passed as an argument.'
        self.assertRaisesRegex(RuntimeError, msg, self.get_conf_path_from_env)

    def test_return_valid_path(self):
        __here__ = os.path.abspath(__file__)
        os.environ['PECAN_CONFIG'] = __here__
        assert self.get_conf_path_from_env() == __here__