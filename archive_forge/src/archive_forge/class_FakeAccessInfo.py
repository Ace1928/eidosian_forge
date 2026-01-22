from unittest import mock
from keystoneauth1 import plugin
class FakeAccessInfo(object):

    def __init__(self, roles, user_domain, project_domain):
        self.roles = roles
        self.user_domain = user_domain
        self.project_domain = project_domain

    @property
    def role_names(self):
        return self.roles

    @property
    def user_domain_id(self):
        return self.user_domain

    @property
    def project_domain_id(self):
        return self.project_domain