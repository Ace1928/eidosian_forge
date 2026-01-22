import copy
import testtools
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.port
def _validate_list(self, expect_request, expect_targets, actual_targets):
    self.assertEqual(expect_request, self.api.calls)
    self.assertEqual(len(expect_targets), len(actual_targets))
    for expect, obj in zip(expect_targets, actual_targets):
        self._validate_obj(expect, obj)