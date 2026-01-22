import copy
import datetime
from unittest import mock
import uuid
from keystoneauth1 import access
from keystoneauth1 import fixture
from osc_lib.cli import format_columns
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit import utils
class TestFederatedIdentity(utils.TestCommand):

    def setUp(self):
        super(TestFederatedIdentity, self).setUp()
        self.app.client_manager.identity = FakeFederatedClient(endpoint=fakes.AUTH_URL, token=fakes.AUTH_TOKEN)