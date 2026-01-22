import os
import tempfile
import unittest
from webtest import TestApp
import pecan
from pecan.tests import PecanTestCase
class TestConfigCleanup(unittest.TestCase):

    def setUp(self):

        class RootController(object):

            @pecan.expose()
            def index(self):
                return 'Hello, World!'
        self.app = TestApp(pecan.Pecan(RootController()))

    def tearDown(self):
        pecan.configuration.set_config(pecan.configuration.DEFAULT, overwrite=True)

    def test_conf_default(self):
        assert pecan.conf.server.to_dict() == {'port': '8080', 'host': '0.0.0.0'}

    def test_conf_changed(self):
        pecan.conf.server = pecan.configuration.Config({'port': '80'})
        assert pecan.conf.server.to_dict() == {'port': '80'}