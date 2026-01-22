import copy
from unittest import mock
from osc_lib.tests import utils
from octaviaclient.tests import fakes
from octaviaclient.tests.unit.osc.v2 import constants
class TestOctaviaClient(utils.TestCommand):

    def setUp(self):
        super().setUp()
        self.app.client_manager.load_balancer = FakeOctaviaClient(endpoint=fakes.AUTH_URL, token=fakes.AUTH_TOKEN)