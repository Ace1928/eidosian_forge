from unittest import mock
import netaddr
import testtools
from neutron_lib.api import converters
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib.tests import _base as base
from neutron_lib.tests import tools
def test_unknown_string(self):
    with testtools.ExpectedException(n_exc.InvalidInput):
        converters.convert_to_protocol('Invalid')