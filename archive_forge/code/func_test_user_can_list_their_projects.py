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
def test_user_can_list_their_projects(self):
    with self.test_client() as c:
        r = c.get('/v3/users/%s/projects' % self.user_id, headers=self.headers)
        self.assertEqual(1, len(r.json['projects']))
        self.assertEqual(self.project_id, r.json['projects'][0]['id'])