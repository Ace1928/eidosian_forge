import string
from unittest import mock
import netaddr
from neutron_lib._i18n import _
from neutron_lib.api import converters
from neutron_lib.api.definitions import extra_dhcp_opt
from neutron_lib.api import validators
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib import fixture
from neutron_lib.plugins import directory
from neutron_lib.tests import _base as base
def test_validate_values_display(self):
    msg = validators.validate_values(7, [4, 6], valid_values_display='[4, 6]')
    self.assertEqual('7 is not in [4, 6]', msg)
    msg = validators.validate_values(7, (4, 6), valid_values_display='(4, 6)')
    self.assertEqual('7 is not in (4, 6)', msg)
    msg = validators.validate_values(8, range(8), valid_values_display='[0..7]')
    self.assertEqual('8 is not in [0..7]', msg)
    msg = validators.validate_values(1, [2, 3, 4, 5], valid_values_display='[2, 3, 4, 5]')
    self.assertEqual('1 is not in [2, 3, 4, 5]', msg)
    msg = validators.validate_values('1', ['2', '3', '4', '5'], valid_values_display="'valid_values_to_show'")
    self.assertEqual("1 is not in 'valid_values_to_show'", msg)
    data = 1
    valid_values = '[2, 3, 4, 5]'
    response = "'data' of type '%s' and 'valid_values' of type '%s' are not compatible for comparison" % (type(data), type(valid_values))
    self.assertRaisesRegex(TypeError, response, validators.validate_values, data, valid_values, valid_values_display='[2, 3, 4, 5]')