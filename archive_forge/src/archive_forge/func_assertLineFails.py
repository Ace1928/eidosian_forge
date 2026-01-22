import testtools
from neutron_lib.hacking import checks
from neutron_lib.hacking import translation_checks as tc
from neutron_lib.tests import _base as base
def assertLineFails(self, func, *args):
    self.assertIsInstance(next(func(*args)), tuple)