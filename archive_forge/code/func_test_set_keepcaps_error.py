from unittest import mock
from oslotest import base
from oslo_privsep import capabilities
@mock.patch('oslo_privsep.capabilities._prctl')
def test_set_keepcaps_error(self, mock_prctl):
    mock_prctl.return_value = -1
    self.assertRaises(OSError, capabilities.set_keepcaps, True)