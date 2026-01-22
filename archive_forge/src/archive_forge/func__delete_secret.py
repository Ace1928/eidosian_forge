import copy
from oslo_serialization import jsonutils
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
import testtools
import barbicanclient.barbican
def _delete_secret(self, auth_version):
    ref = '{0}/secrets/{1}'.format(BARBICAN_ENDPOINT, uuidutils.generate_uuid())
    self.responses.delete(ref, status_code=204)
    argv = self.get_arguments(auth_version)
    argv.extend(['--endpoint', BARBICAN_ENDPOINT, 'secret', 'delete', ref])
    try:
        self.barbican.run(argv=argv)
    except Exception:
        self.fail('failed to delete secret')