from unittest import mock
from glance.domain import proxy
import glance.tests.utils as test_utils
def test_remove_with_no_result(self):
    self._test_method_with_proxied_argument('remove', None)