import copy
from oslo_serialization import jsonutils
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
import testtools
import barbicanclient.barbican
class KeystoneClientFixture(testtools.TestCase):

    def setUp(self):
        super(KeystoneClientFixture, self).setUp()
        self.responses = self.useFixture(fixture.Fixture())
        self.barbican = barbicanclient.barbican.Barbican()
        self.test_arguments = {}

    def get_arguments(self, auth_version='v3'):
        if auth_version.lower() == 'v3':
            version_specific = {'--os-auth-url': V3_URL, '--os-project-name': 'my_project_name'}
        else:
            version_specific = {'--os-auth-url': V2_URL, '--os-identity-api-version': '2.0', '--os-tenant-name': 'my_tenant_name'}
        self.test_arguments.update(version_specific)
        return self._to_argv(self.test_arguments)

    def _to_argv(self, argument_dict):
        argv = []
        for k, v in argument_dict.items():
            argv.extend([k, v])
        return argv

    def _delete_secret(self, auth_version):
        ref = '{0}/secrets/{1}'.format(BARBICAN_ENDPOINT, uuidutils.generate_uuid())
        self.responses.delete(ref, status_code=204)
        argv = self.get_arguments(auth_version)
        argv.extend(['--endpoint', BARBICAN_ENDPOINT, 'secret', 'delete', ref])
        try:
            self.barbican.run(argv=argv)
        except Exception:
            self.fail('failed to delete secret')

    def test_v2_auth(self):
        self.responses.get(V2_URL, body=V2_VERSION_ENTRY)
        self.responses.post('{0}/tokens'.format(V2_URL), json=generate_v2_project_scoped_token())
        self._delete_secret('v2')

    def test_v3_auth(self):
        self.responses.get(V3_URL, text=V3_VERSION_ENTRY)
        id, v3_token = generate_v3_project_scoped_token()
        self.responses.post('{0}/auth/tokens'.format(V3_URL), json=v3_token, headers={'X-Subject-Token': '1234'})
        self._delete_secret('v3')