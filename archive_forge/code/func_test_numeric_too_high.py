from unittest import mock
import netaddr
import testtools
from neutron_lib.api import converters
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib.tests import _base as base
from neutron_lib.tests import tools
def test_numeric_too_high(self):
    with testtools.ExpectedException(n_exc.InvalidInput):
        converters.convert_to_protocol('300')