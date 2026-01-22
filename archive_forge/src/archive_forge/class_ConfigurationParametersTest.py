import json
import testtools
from unittest import mock
from troveclient import base
from troveclient.v1 import configurations
from troveclient.v1 import management
class ConfigurationParametersTest(testtools.TestCase):

    def setUp(self):
        super(ConfigurationParametersTest, self).setUp()
        self.orig__init = configurations.ConfigurationParameters.__init__
        configurations.ConfigurationParameters.__init__ = mock.Mock(return_value=None)
        self.config_params = configurations.ConfigurationParameters()
        self.config_params.api = mock.Mock()
        self.config_params.api.client = mock.Mock()

    def tearDown(self):
        super(ConfigurationParametersTest, self).tearDown()
        configurations.ConfigurationParameters.__init__ = self.orig__init

    def test_list_parameters(self):

        def side_effect_func(path, config):
            return path
        self.config_params._list = mock.Mock(side_effect=side_effect_func)
        self.assertEqual('/datastores/datastore/versions/version/parameters', self.config_params.parameters('datastore', 'version'))

    def test_get_parameter(self):

        def side_effect_func(path):
            return path
        self.config_params._get = mock.Mock(side_effect=side_effect_func)
        self.assertEqual('/datastores/datastore/versions/version/parameters/key', self.config_params.get_parameter('datastore', 'version', 'key'))

    def test_list_parameters_by_version(self):

        def side_effect_func(path, config):
            return path
        self.config_params._list = mock.Mock(side_effect=side_effect_func)
        self.assertEqual('/datastores/versions/version/parameters', self.config_params.parameters_by_version('version'))

    def test_get_parameter_by_version(self):

        def side_effect_func(path):
            return path
        self.config_params._get = mock.Mock(side_effect=side_effect_func)
        self.assertEqual('/datastores/versions/version/parameters/key', self.config_params.get_parameter_by_version('version', 'key'))