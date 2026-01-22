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
class _AssignmentTestUtilities(object):
    """Useful utilities for setting up test assignments and assertions."""

    def _setup_test_role_assignments(self):
        role_id = self.bootstrapper.reader_role_id
        user = PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=CONF.identity.default_domain_id))
        group = PROVIDERS.identity_api.create_group(unit.new_group_ref(domain_id=CONF.identity.default_domain_id))
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=CONF.identity.default_domain_id))
        PROVIDERS.assignment_api.create_grant(role_id, user_id=user['id'], project_id=project['id'])
        PROVIDERS.assignment_api.create_grant(role_id, user_id=user['id'], domain_id=domain['id'])
        PROVIDERS.assignment_api.create_system_grant_for_user(user['id'], role_id)
        PROVIDERS.assignment_api.create_grant(role_id, group_id=group['id'], project_id=project['id'])
        PROVIDERS.assignment_api.create_grant(role_id, group_id=group['id'], domain_id=domain['id'])
        PROVIDERS.assignment_api.create_system_grant_for_group(group['id'], role_id)
        return {'user_id': user['id'], 'group_id': group['id'], 'domain_id': domain['id'], 'project_id': project['id'], 'role_id': role_id}

    def _extract_role_assignments_from_response_body(self, r):
        assignments = []
        for assignment in r.json['role_assignments']:
            a = {}
            if 'project' in assignment['scope']:
                a['project_id'] = assignment['scope']['project']['id']
            elif 'domain' in assignment['scope']:
                a['domain_id'] = assignment['scope']['domain']['id']
            elif 'system' in assignment['scope']:
                a['system'] = 'all'
            if 'user' in assignment:
                a['user_id'] = assignment['user']['id']
            elif 'group' in assignment:
                a['group_id'] = assignment['group']['id']
            a['role_id'] = assignment['role']['id']
            assignments.append(a)
        return assignments