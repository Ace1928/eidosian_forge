import copy
import testtools
from testtools import matchers
from urllib import parse
from zunclient.common import utils as zun_utils
from zunclient import exceptions
from zunclient.tests.unit import utils
from zunclient.v1 import containers
def _test_containers_list_with_filters(self, limit=None, marker=None, sort_key=None, sort_dir=None, expect=[]):
    containers_filter = self.mgr.list(limit=limit, marker=marker, sort_key=sort_key, sort_dir=sort_dir)
    self.assertEqual(expect, self.api.calls)
    self.assertThat(containers_filter, matchers.HasLength(2))