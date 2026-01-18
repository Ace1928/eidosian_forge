import inspect
import re
import socket
import sys
from .. import controldir, errors, osutils, tests, urlutils
def test_socket_connection_error(self):
    """Test the formatting of SocketConnectionError"""
    self.assertSocketConnectionError('Failed to connect to ahost', 'ahost')
    self.assertSocketConnectionError('Failed to connect to ahost', 'ahost', port=None)
    self.assertSocketConnectionError('Failed to connect to ahost:22', 'ahost', port=22)
    self.assertSocketConnectionError('Failed to connect to ahost:22; bogus error', 'ahost', port=22, orig_error='bogus error')
    self.assertSocketConnectionError('Failed to connect to ahost; bogus error', 'ahost', orig_error='bogus error')
    orig_error = ValueError('bad value')
    self.assertSocketConnectionError('Failed to connect to ahost; {}'.format(str(orig_error)), host='ahost', orig_error=orig_error)
    self.assertSocketConnectionError('Unable to connect to ssh host ahost:444; my_error', host='ahost', port=444, msg='Unable to connect to ssh host', orig_error='my_error')