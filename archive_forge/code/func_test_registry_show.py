import copy
import testtools
from testtools import matchers
from zunclient import exceptions
from zunclient.tests.unit import utils
from zunclient.v1 import registries
def test_registry_show(self):
    registry = self.mgr.get(REGISTRY1['uuid'])
    expect = [('GET', '/v1/registries/%s' % REGISTRY1['uuid'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(REGISTRY1['name'], registry._info['registry']['name'])
    self.assertEqual(REGISTRY1['uuid'], registry._info['registry']['uuid'])