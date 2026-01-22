import datetime
import random
import uuid
import freezegun
import http.client
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.resource.backends import base as resource_base
from keystone.tests import unit
from keystone.tests.unit import test_v3
class ListUserProjectsTestCase(test_v3.RestfulTestCase):
    """Test for /users/<user>/projects."""

    def load_sample_data(self):
        self.auths = []
        self.domains = []
        self.projects = []
        self.roles = []
        self.users = []
        root_domain = unit.new_domain_ref(id=resource_base.NULL_DOMAIN_ID, name=resource_base.NULL_DOMAIN_ID)
        self.resource_api.create_domain(resource_base.NULL_DOMAIN_ID, root_domain)
        for _ in range(3):
            domain = unit.new_domain_ref()
            PROVIDERS.resource_api.create_domain(domain['id'], domain)
            user = unit.create_user(PROVIDERS.identity_api, domain_id=domain['id'])
            role = unit.new_role_ref()
            PROVIDERS.role_api.create_role(role['id'], role)
            PROVIDERS.assignment_api.create_grant(role['id'], user_id=user['id'], domain_id=domain['id'])
            project = unit.new_project_ref(domain_id=domain['id'])
            PROVIDERS.resource_api.create_project(project['id'], project)
            PROVIDERS.assignment_api.create_grant(role['id'], user_id=user['id'], project_id=project['id'])
            auth = self.build_authentication_request(user_id=user['id'], password=user['password'], domain_id=domain['id'])
            self.auths.append(auth)
            self.domains.append(domain)
            self.projects.append(project)
            self.roles.append(role)
            self.users.append(user)

    def test_list_head_all(self):
        for i in range(len(self.users)):
            user = self.users[i]
            auth = self.auths[i]
            url = '/users/%s/projects' % user['id']
            result = self.get(url, auth=auth)
            projects_result = result.json['projects']
            self.assertEqual(1, len(projects_result))
            self.assertEqual(self.projects[i]['id'], projects_result[0]['id'])
            self.head(url, auth=auth, expected_status=http.client.OK)

    def test_list_enabled(self):
        for i in range(len(self.users)):
            user = self.users[i]
            auth = self.auths[i]
            url = '/users/%s/projects?enabled=True' % user['id']
            result = self.get(url, auth=auth)
            projects_result = result.json['projects']
            self.assertEqual(1, len(projects_result))
            self.assertEqual(self.projects[i]['id'], projects_result[0]['id'])

    def test_list_disabled(self):
        for i in range(len(self.users)):
            user = self.users[i]
            auth = self.auths[i]
            project = self.projects[i]
            url = '/users/%s/projects?enabled=False' % user['id']
            result = self.get(url, auth=auth)
            self.assertEqual(0, len(result.json['projects']))
            project['enabled'] = False
            PROVIDERS.resource_api.update_project(project['id'], project)
            result = self.get(url, auth=auth)
            projects_result = result.json['projects']
            self.assertEqual(1, len(projects_result))
            self.assertEqual(self.projects[i]['id'], projects_result[0]['id'])

    def test_list_by_domain_id(self):
        for i in range(len(self.users)):
            user = self.users[i]
            domain = self.domains[i]
            auth = self.auths[i]
            url = '/users/%s/projects?domain_id=%s' % (user['id'], uuid.uuid4().hex)
            result = self.get(url, auth=auth)
            self.assertEqual(0, len(result.json['projects']))
            url = '/users/%s/projects?domain_id=%s' % (user['id'], domain['id'])
            result = self.get(url, auth=auth)
            projects_result = result.json['projects']
            self.assertEqual(1, len(projects_result))
            self.assertEqual(self.projects[i]['id'], projects_result[0]['id'])