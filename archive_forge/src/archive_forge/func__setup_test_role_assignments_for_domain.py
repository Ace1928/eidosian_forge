import copy
import http.client
import uuid
from oslo_serialization import jsonutils
from keystone.common.policies import role_assignment as rp
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
def _setup_test_role_assignments_for_domain(self):
    role_id = self.bootstrapper.reader_role_id
    user = PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=CONF.identity.default_domain_id))
    group = PROVIDERS.identity_api.create_group(unit.new_group_ref(domain_id=CONF.identity.default_domain_id))
    project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=self.domain_id))
    PROVIDERS.assignment_api.create_grant(role_id, user_id=user['id'], project_id=project['id'])
    PROVIDERS.assignment_api.create_grant(role_id, user_id=user['id'], domain_id=self.domain_id)
    PROVIDERS.assignment_api.create_grant(role_id, group_id=group['id'], project_id=project['id'])
    PROVIDERS.assignment_api.create_grant(role_id, group_id=group['id'], domain_id=self.domain_id)
    return {'user_id': user['id'], 'group_id': group['id'], 'project_id': project['id'], 'role_id': role_id}