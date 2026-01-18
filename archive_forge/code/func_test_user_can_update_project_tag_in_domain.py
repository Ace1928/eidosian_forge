import uuid
import http.client
from oslo_serialization import jsonutils
from keystone.common.policies import project as pp
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
def test_user_can_update_project_tag_in_domain(self):
    project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=self.domain_id))
    tag = uuid.uuid4().hex
    PROVIDERS.resource_api.create_project_tag(project['id'], tag)
    new_tag = uuid.uuid4().hex
    update = {'tags': [new_tag]}
    with self.test_client() as c:
        r = c.put('/v3/projects/%s/tags' % project['id'], headers=self.headers, json=update, expected_status_code=http.client.OK)
        self.assertTrue(len(r.json['tags']) == 1)
        self.assertEqual(new_tag, r.json['tags'][0])