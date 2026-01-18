import copy
import testtools
from testtools import matchers
from zunclient import exceptions
from zunclient.tests.unit import utils
from zunclient.v1 import registries
def test_registry_update(self):
    registry = self.mgr.update(REGISTRY1['uuid'], **UPDATE_REGISTRY1)
    expect = [('PATCH', '/v1/registries/%s' % REGISTRY1['uuid'], {}, {'registry': UPDATE_REGISTRY1})]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(UPDATE_REGISTRY1['name'], registry._info['registry']['name'])