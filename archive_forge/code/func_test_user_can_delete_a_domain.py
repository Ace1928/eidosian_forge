import uuid
import http.client
from oslo_serialization import jsonutils
from keystone.common.policies import domain as dp
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
def test_user_can_delete_a_domain(self):
    domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
    with self.test_client() as c:
        update = {'domain': {'enabled': False}}
        path = '/v3/domains/%s' % domain['id']
        c.patch(path, json=update, headers=self.headers)
        c.delete(path, headers=self.headers)