import json
import testtools
from unittest import mock
from troveclient import base
from troveclient.v1 import configurations
from troveclient.v1 import management
class ConfigurationTest(testtools.TestCase):

    def setUp(self):
        super(ConfigurationTest, self).setUp()
        self.orig__init = configurations.Configuration.__init__
        configurations.Configuration.__init__ = mock.Mock(return_value=None)
        self.configuration = configurations.Configuration()

    def tearDown(self):
        super(ConfigurationTest, self).tearDown()
        configurations.Configuration.__init__ = self.orig__init

    def test___repr__(self):
        self.configuration.name = 'config-1'
        self.assertEqual('<Configuration: config-1>', self.configuration.__repr__())