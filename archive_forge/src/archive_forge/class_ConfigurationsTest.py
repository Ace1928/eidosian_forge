import json
import testtools
from unittest import mock
from troveclient import base
from troveclient.v1 import configurations
from troveclient.v1 import management
class ConfigurationsTest(testtools.TestCase):

    def setUp(self):
        super(ConfigurationsTest, self).setUp()
        self.orig__init = configurations.Configurations.__init__
        configurations.Configurations.__init__ = mock.Mock(return_value=None)
        self.configurations = configurations.Configurations()
        self.configurations.api = mock.Mock()
        self.configurations.api.client = mock.Mock()
        self.orig_base_getid = base.getid
        base.getid = mock.Mock(return_value='configuration1')

    def tearDown(self):
        super(ConfigurationsTest, self).tearDown()
        configurations.Configurations.__init__ = self.orig__init
        base.getid = self.orig_base_getid

    def _get_mock_method(self):
        self._resp = mock.Mock()
        self._body = None
        self._url = None

        def side_effect_func(url, body=None):
            self._body = body
            self._url = url
            return (self._resp, body)
        return mock.Mock(side_effect=side_effect_func)

    def test_create(self):
        self.configurations.api.client.post = self._get_mock_method()
        self._resp.status_code = 200
        config = '{"test":12}'
        self.configurations.create('config1', config, 'test description')
        self.assertEqual('/configurations', self._url)
        expected = {'description': 'test description', 'name': 'config1', 'values': json.loads(config)}
        self.assertEqual({'configuration': expected}, self._body)

    def test_delete(self):
        self.configurations.api.client.delete = self._get_mock_method()
        self._resp.status_code = 200
        self.configurations.delete(27)
        self.assertEqual('/configurations/configuration1', self._url)
        self._resp.status_code = 500
        self.assertRaises(Exception, self.configurations.delete, 34)

    def test_list(self):
        page_mock = mock.Mock()
        self.configurations._paginated = page_mock
        limit = 'test-limit'
        marker = 'test-marker'
        self.configurations.list(limit, marker)
        page_mock.assert_called_with('/configurations', 'configurations', limit, marker)

    def test_get(self):

        def side_effect_func(path, config):
            return (path, config)
        self.configurations._get = mock.Mock(side_effect=side_effect_func)
        self.assertEqual(('/configurations/configuration1', 'configuration'), self.configurations.get(123))

    def test_instances(self):
        page_mock = mock.Mock()
        self.configurations._paginated = page_mock
        limit = 'test-limit'
        marker = 'test-marker'
        configuration = 'configuration1'
        self.configurations.instances(configuration, limit, marker)
        page_mock.assert_called_with('/configurations/' + configuration + '/instances', 'instances', limit, marker)

    def test_update(self):
        self.configurations.api.client.put = self._get_mock_method()
        self._resp.status_code = 200
        config = '{"test":12}'
        self.configurations.update(27, config)
        self.assertEqual('/configurations/configuration1', self._url)
        self._resp.status_code = 500
        self.assertRaises(Exception, self.configurations.update, 34)

    def test_edit(self):
        self.configurations.api.client.patch = self._get_mock_method()
        self._resp.status_code = 200
        config = '{"test":12}'
        self.configurations.edit(27, config)
        self.assertEqual('/configurations/configuration1', self._url)
        self._resp.status_code = 500
        self.assertRaises(Exception, self.configurations.edit, 34)