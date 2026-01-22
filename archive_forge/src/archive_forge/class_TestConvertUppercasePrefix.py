from unittest import mock
import netaddr
import testtools
from neutron_lib.api import converters
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib.tests import _base as base
from neutron_lib.tests import tools
class TestConvertUppercasePrefix(base.BaseTestCase):

    def test_prefix_not_present(self):
        self.assertEqual('foobar', converters.convert_prefix_forced_case('foobar', 'bar'))

    def test_prefix_no_need_to_replace(self):
        self.assertEqual('FOObar', converters.convert_prefix_forced_case('FOObar', 'FOO'))

    def test_ucfirst_prefix_converted_1(self):
        self.assertEqual('Foobar', converters.convert_prefix_forced_case('foobar', 'Foo'))

    def test_lc_prefix_converted_2(self):
        self.assertEqual('foobar', converters.convert_prefix_forced_case('fOobar', 'foo'))

    def test_mixed_prefix_converted_1(self):
        self.assertEqual('fOoXbar', converters.convert_prefix_forced_case('Fooxbar', 'fOoX'))

    def test_shorter_string(self):
        self.assertEqual('fo', converters.convert_prefix_forced_case('fo', 'foo'))