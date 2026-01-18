from unittest import mock
import ddt
from manilaclient import exceptions
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import security_services
def test_update_no_fields_specified(self):
    security_service = 'fake service'
    self.assertRaises(exceptions.CommandError, self.manager.update, security_service)