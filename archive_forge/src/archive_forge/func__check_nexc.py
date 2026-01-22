import functools
from neutron_lib._i18n import _
import neutron_lib.exceptions as ne
from neutron_lib.tests import _base as base
def _check_nexc(self, exc_class, expected_msg, **kwargs):
    raise_exc_class = functools.partial(_raise, exc_class)
    e = self.assertRaises(exc_class, raise_exc_class, **kwargs)
    self.assertEqual(expected_msg, str(e))