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
def test_user_cannot_get_tag_for_project_outside_domain(self):
    project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=CONF.identity.default_domain_id))
    tag = uuid.uuid4().hex
    PROVIDERS.resource_api.create_project_tag(project['id'], tag)
    with self.test_client() as c:
        c.get('/v3/projects/%s/tags/%s' % (project['id'], tag), headers=self.headers, expected_status_code=http.client.FORBIDDEN)