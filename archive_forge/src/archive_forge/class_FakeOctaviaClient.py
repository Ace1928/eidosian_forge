import copy
from unittest import mock
from osc_lib.tests import utils
from octaviaclient.tests import fakes
from octaviaclient.tests.unit.osc.v2 import constants
class FakeOctaviaClient(object):

    def __init__(self, **kwargs):
        self.load_balancers = mock.Mock()
        self.load_balancers.resource_class = fakes.FakeResource(None, {})
        self.auth_token = kwargs['token']
        self.management_url = kwargs['endpoint']