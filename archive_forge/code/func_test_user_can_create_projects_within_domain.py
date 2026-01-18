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
def test_user_can_create_projects_within_domain(self):
    create = {'project': unit.new_project_ref(domain_id=self.domain_id)}
    with self.test_client() as c:
        c.post('/v3/projects', json=create, headers=self.headers)