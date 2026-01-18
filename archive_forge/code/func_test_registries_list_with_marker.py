import copy
import testtools
from testtools import matchers
from zunclient import exceptions
from zunclient.tests.unit import utils
from zunclient.v1 import registries
def test_registries_list_with_marker(self):
    expect = [('GET', '/v1/registries/?marker=%s' % REGISTRY2['uuid'], {}, None)]
    self._test_registries_list_with_filters(marker=REGISTRY2['uuid'], expect=expect)