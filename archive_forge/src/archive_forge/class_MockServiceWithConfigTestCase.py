from boto.compat import http_client
from tests.compat import mock, unittest
class MockServiceWithConfigTestCase(AWSMockServiceTestCase):

    def setUp(self):
        super(MockServiceWithConfigTestCase, self).setUp()
        self.environ = {}
        self.config = {}
        self.config_patch = mock.patch('boto.provider.config.get', self.get_config)
        self.has_config_patch = mock.patch('boto.provider.config.has_option', self.has_config)
        self.environ_patch = mock.patch('os.environ', self.environ)
        self.config_patch.start()
        self.has_config_patch.start()
        self.environ_patch.start()

    def tearDown(self):
        self.config_patch.stop()
        self.has_config_patch.stop()
        self.environ_patch.stop()

    def has_config(self, section_name, key):
        try:
            self.config[section_name][key]
            return True
        except KeyError:
            return False

    def get_config(self, section_name, key, default=None):
        try:
            return self.config[section_name][key]
        except KeyError:
            return None