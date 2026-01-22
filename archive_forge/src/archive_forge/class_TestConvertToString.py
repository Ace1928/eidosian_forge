from unittest import mock
import netaddr
import testtools
from neutron_lib.api import converters
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib.tests import _base as base
from neutron_lib.tests import tools
class TestConvertToString(base.BaseTestCase):

    def test_data_is_string(self):
        self.assertEqual('10000', converters.convert_to_string('10000'))

    def test_data_is_integer(self):
        self.assertEqual('10000', converters.convert_to_string(10000))

    def test_data_is_integer_zero(self):
        self.assertEqual('0', converters.convert_to_string(0))

    def test_data_is_none(self):
        self.assertIsNone(converters.convert_to_string(None))

    def test_data_is_empty_list(self):
        self.assertEqual('[]', converters.convert_to_string([]))

    def test_data_is_list(self):
        self.assertEqual('[1, 2, 3]', converters.convert_to_string([1, 2, 3]))

    def test_data_is_empty_dict(self):
        self.assertEqual('{}', converters.convert_to_string({}))

    def test_data_is_dict(self):
        self.assertEqual("{'foo': 'bar'}", converters.convert_to_string({'foo': 'bar'}))