import functools
from neutron_lib._i18n import _
import neutron_lib.exceptions as ne
from neutron_lib.tests import _base as base
def test_resource_exhausted(self):
    self._check_nexc(ne.ResourceExhausted, _('The service is unavailable.'))