from unittest import mock
from keystoneauth1 import plugin
class FakeAuth(plugin.BaseAuthPlugin):

    def __init__(self, auth_token='abcd1234', only_services=None):
        self.auth_token = auth_token
        self.only_services = only_services

    def get_token(self, session, **kwargs):
        return self.auth_token

    def get_endpoint(self, session, service_type=None, **kwargs):
        if self.only_services is not None and service_type not in self.only_services:
            return None
        return 'http://example.com:1234/v1'

    def get_auth_ref(self, session):
        return mock.Mock()

    def get_access(self, sesssion):
        return FakeAccessInfo([], None, None)