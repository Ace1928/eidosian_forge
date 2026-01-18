import functools
from neutron_lib._i18n import _
import neutron_lib.exceptions as ne
from neutron_lib.tests import _base as base
def test_policy_check_error(self):
    self._check_nexc(ne.PolicyCheckError, _('Failed to check policy policy because reason.'), policy='policy', reason='reason')