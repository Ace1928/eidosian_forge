from keystoneclient.tests.unit.v2_0 import utils
from keystoneclient.v2_0 import extensions
class ExtensionTests(utils.ClientTestCase):

    def setUp(self):
        super(ExtensionTests, self).setUp()
        self.TEST_EXTENSIONS = {'extensions': {'values': [{'name': 'OpenStack Keystone User CRUD', 'namespace': 'https://docs.openstack.org/identity/api/ext/OS-KSCRUD/v1.0', 'updated': '2013-07-07T12:00:0-00:00', 'alias': 'OS-KSCRUD', 'description': 'OpenStack extensions to Keystone v2.0 API enabling User Operations.', 'links': '[{"href":"https://github.com/openstack/identity-api", "type": "text/html", "rel": "describedby"}]'}, {'name': 'OpenStack EC2 API', 'namespace': 'https://docs.openstack.org/identity/api/ext/OS-EC2/v1.0', 'updated': '2013-09-07T12:00:0-00:00', 'alias': 'OS-EC2', 'description': 'OpenStack EC2 Credentials backend.', 'links': '[{"href":"https://github.com/openstack/identity-api", "type": "text/html", "rel": "describedby"}]'}]}}

    def test_list(self):
        self.stub_url('GET', ['extensions'], json=self.TEST_EXTENSIONS)
        extensions_list = self.client.extensions.list()
        self.assertEqual(2, len(extensions_list))
        for extension in extensions_list:
            self.assertIsInstance(extension, extensions.Extension)
            self.assertIsNotNone(extension.alias)
            self.assertIsNotNone(extension.description)
            self.assertIsNotNone(extension.links)
            self.assertIsNotNone(extension.name)
            self.assertIsNotNone(extension.namespace)
            self.assertIsNotNone(extension.updated)