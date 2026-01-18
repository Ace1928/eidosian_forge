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
def test_validate_dict_with_invalid_validator(self):
    dictionary, constraints = self._construct_dict_and_constraints()
    constraints['key1'] = {'type:unsupported': None, 'required': True}
    msg = validators.validate_dict(dictionary, constraints)
    self.assertEqual("Validator 'type:unsupported' does not exist", msg)