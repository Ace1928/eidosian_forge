import copy
import random
from unittest import mock
import uuid
from openstack.image.v1 import _proxy as image_v1_proxy
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
from openstackclient.tests.unit import utils
class TestVolumev1(FakeClientMixin, utils.TestCommand):

    def setUp(self):
        super().setUp()
        self.app.client_manager.identity = identity_fakes.FakeIdentityv2Client(endpoint=fakes.AUTH_URL, token=fakes.AUTH_TOKEN)
        self.app.client_manager.image = mock.Mock(spec=image_v1_proxy.Proxy)
        self.image_client = self.app.client_manager.image