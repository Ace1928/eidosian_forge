import copy
from unittest import mock
import uuid
from keystoneauth1 import access
from keystoneauth1 import fixture
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit import utils
class FakeIdentityv2Client(object):

    def __init__(self, **kwargs):
        self.roles = mock.Mock()
        self.roles.resource_class = fakes.FakeResource(None, {})
        self.services = mock.Mock()
        self.services.resource_class = fakes.FakeResource(None, {})
        self.tenants = mock.Mock()
        self.tenants.resource_class = fakes.FakeResource(None, {})
        self.tokens = mock.Mock()
        self.tokens.resource_class = fakes.FakeResource(None, {})
        self.users = mock.Mock()
        self.users.resource_class = fakes.FakeResource(None, {})
        self.ec2 = mock.Mock()
        self.ec2.resource_class = fakes.FakeResource(None, {})
        self.endpoints = mock.Mock()
        self.endpoints.resource_class = fakes.FakeResource(None, {})
        self.extensions = mock.Mock()
        self.extensions.resource_class = fakes.FakeResource(None, {})
        self.auth_token = kwargs['token']
        self.management_url = kwargs['endpoint']

    def __getattr__(self, name):
        if name == 'projects':
            return self.tenants
        else:
            raise AttributeError(name)