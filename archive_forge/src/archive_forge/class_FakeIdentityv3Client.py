import copy
import datetime
from unittest import mock
import uuid
from keystoneauth1 import access
from keystoneauth1 import fixture
from osc_lib.cli import format_columns
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit import utils
class FakeIdentityv3Client(object):

    def __init__(self, **kwargs):
        self.domains = mock.Mock()
        self.domains.resource_class = fakes.FakeResource(None, {})
        self.credentials = mock.Mock()
        self.credentials.resource_class = fakes.FakeResource(None, {})
        self.endpoints = mock.Mock()
        self.endpoints.resource_class = fakes.FakeResource(None, {})
        self.endpoint_filter = mock.Mock()
        self.endpoint_filter.resource_class = fakes.FakeResource(None, {})
        self.endpoint_groups = mock.Mock()
        self.endpoint_groups.resource_class = fakes.FakeResource(None, {})
        self.groups = mock.Mock()
        self.groups.resource_class = fakes.FakeResource(None, {})
        self.oauth1 = mock.Mock()
        self.oauth1.resource_class = fakes.FakeResource(None, {})
        self.projects = mock.Mock()
        self.projects.resource_class = fakes.FakeResource(None, {})
        self.regions = mock.Mock()
        self.regions.resource_class = fakes.FakeResource(None, {})
        self.roles = mock.Mock()
        self.roles.resource_class = fakes.FakeResource(None, {})
        self.services = mock.Mock()
        self.services.resource_class = fakes.FakeResource(None, {})
        self.session = mock.Mock()
        self.session.auth.auth_ref.service_catalog.resource_class = fakes.FakeResource(None, {})
        self.tokens = mock.Mock()
        self.tokens.resource_class = fakes.FakeResource(None, {})
        self.trusts = mock.Mock()
        self.trusts.resource_class = fakes.FakeResource(None, {})
        self.users = mock.Mock()
        self.users.resource_class = fakes.FakeResource(None, {})
        self.role_assignments = mock.Mock()
        self.role_assignments.resource_class = fakes.FakeResource(None, {})
        self.auth_token = kwargs['token']
        self.management_url = kwargs['endpoint']
        self.auth = FakeAuth()
        self.auth.client = mock.Mock()
        self.auth.client.resource_class = fakes.FakeResource(None, {})
        self.application_credentials = mock.Mock()
        self.application_credentials.resource_class = fakes.FakeResource(None, {})
        self.access_rules = mock.Mock()
        self.access_rules.resource_class = fakes.FakeResource(None, {})
        self.inference_rules = mock.Mock()
        self.inference_rules.resource_class = fakes.FakeResource(None, {})
        self.registered_limits = mock.Mock()
        self.registered_limits.resource_class = fakes.FakeResource(None, {})
        self.limits = mock.Mock()
        self.limits.resource_class = fakes.FakeResource(None, {})