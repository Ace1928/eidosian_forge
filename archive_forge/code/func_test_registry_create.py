import copy
import testtools
from testtools import matchers
from zunclient import exceptions
from zunclient.tests.unit import utils
from zunclient.v1 import registries
def test_registry_create(self):
    registries = self.mgr.create(**CREATE_REGISTRY1)
    expect = [('POST', '/v1/registries', {}, {'registry': CREATE_REGISTRY1})]
    self.assertEqual(expect, self.api.calls)
    self.assertTrue(registries)