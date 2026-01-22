from unittest import mock
from openstackclient.identity.v2_0 import token
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
class TestToken(identity_fakes.TestIdentityv2):
    fake_user = identity_fakes.FakeUser.create_one_user()
    fake_project = identity_fakes.FakeProject.create_one_project()

    def setUp(self):
        super(TestToken, self).setUp()
        self.ar_mock = mock.PropertyMock()
        type(self.app.client_manager).auth_ref = self.ar_mock