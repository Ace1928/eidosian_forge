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
class FakeInstances(object):
    fake_instances = fakes.FakeHTTPClient().get_instances()[2]['instances']
    fake_instance = fakes.FakeHTTPClient().get_instance_create()[2]

    def get_instances_1234(self):
        return instances.Instance(None, self.fake_instances[0])

    def get_instances(self):
        return [instances.Instance(None, fake_instance) for fake_instance in self.fake_instances]

    def get_instance_create(self):
        return instances.Instance(None, self.fake_instance['instance'])