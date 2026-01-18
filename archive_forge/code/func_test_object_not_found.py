import functools
from neutron_lib._i18n import _
import neutron_lib.exceptions as ne
from neutron_lib.tests import _base as base
def test_object_not_found(self):
    self._check_nexc(ne.ObjectNotFound, _('Object fallout tato not found.'), id='fallout tato')