from osc_lib import exceptions
from osc_lib.tests import utils
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import validate
def test_validate_TCP_UDP_SCTP_port_range(self):
    validate._validate_TCP_UDP_SCTP_port_range(constants.MIN_PORT_NUMBER, 'fake-parameter')
    validate._validate_TCP_UDP_SCTP_port_range(constants.MAX_PORT_NUMBER, 'fake-parameter')
    self.assertRaises(exceptions.InvalidValue, validate._validate_TCP_UDP_SCTP_port_range, constants.MIN_PORT_NUMBER - 1, 'fake-parameter')
    self.assertRaises(exceptions.InvalidValue, validate._validate_TCP_UDP_SCTP_port_range, constants.MAX_PORT_NUMBER + 1, 'fake-parameter')