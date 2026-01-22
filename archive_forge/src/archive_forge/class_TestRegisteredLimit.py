import copy
from keystoneauth1.exceptions import http as ksa_exceptions
from osc_lib import exceptions
from openstackclient.identity.v3 import registered_limit
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestRegisteredLimit(identity_fakes.TestIdentityv3):

    def setUp(self):
        super(TestRegisteredLimit, self).setUp()
        identity_manager = self.app.client_manager.identity
        self.registered_limit_mock = identity_manager.registered_limits
        self.services_mock = identity_manager.services
        self.services_mock.reset_mock()
        self.regions_mock = identity_manager.regions
        self.regions_mock.reset_mock()