import copy
import testtools
from testtools import matchers
from zunclient import exceptions
from zunclient.tests.unit import utils
from zunclient.v1 import registries
def test_registries_list(self):
    registries = self.mgr.list()
    expect = [('GET', '/v1/registries', {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertThat(registries, matchers.HasLength(2))