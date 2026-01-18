import functools
from neutron_lib._i18n import _
import neutron_lib.exceptions as ne
from neutron_lib.tests import _base as base
def test_over_quota(self):
    self._check_nexc(ne.OverQuota, _('Quota exceeded for resources: tube socks.'), overs='tube socks')