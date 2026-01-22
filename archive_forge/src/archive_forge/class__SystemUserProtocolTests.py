import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
class _SystemUserProtocolTests(object):
    """Common default functionality for all system users."""

    def test_user_can_list_protocols(self):
        protocol, mapping, identity_provider = self._create_protocol_and_deps()
        with self.test_client() as c:
            path = '/v3/OS-FEDERATION/identity_providers/%s/protocols' % identity_provider['id']
            r = c.get(path, headers=self.headers)
            self.assertEqual(1, len(r.json['protocols']))
            self.assertEqual(protocol['id'], r.json['protocols'][0]['id'])

    def test_user_can_get_a_protocol(self):
        protocol, mapping, identity_provider = self._create_protocol_and_deps()
        with self.test_client() as c:
            path = '/v3/OS-FEDERATION/identity_providers/%s/protocols/%s' % (identity_provider['id'], protocol['id'])
            c.get(path, headers=self.headers)