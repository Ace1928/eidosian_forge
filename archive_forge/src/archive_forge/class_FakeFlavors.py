from unittest import mock
from troveclient.tests import fakes
from troveclient.tests.osc import utils
from troveclient.v1 import backups
from troveclient.v1 import clusters
from troveclient.v1 import configurations
from troveclient.v1 import databases
from troveclient.v1 import datastores
from troveclient.v1 import flavors
from troveclient.v1 import instances
from troveclient.v1 import limits
from troveclient.v1 import modules
from troveclient.v1 import quota
from troveclient.v1 import users
class FakeFlavors(object):
    fake_flavors = fakes.FakeHTTPClient().get_flavors()[2]['flavors']

    def get_flavors_1(self):
        return flavors.Flavor(None, self.fake_flavors[0])