from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import logging
import threading
import time
from googlecloudsdk.api_lib.compute import iap_tunnel_websocket_helper as helper
from googlecloudsdk.api_lib.compute import iap_tunnel_websocket_utils as utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import retry
import six
from six.moves import queue
def _WaitForOpenOrRaiseError(self):
    """Wait for WebSocket open confirmation or any error condition."""
    for _ in range(MAX_WEBSOCKET_OPEN_WAIT_TIME_SEC * 100):
        if self._IsClosed():
            break
        if self._HasConnected():
            return
        time.sleep(0.01)
    if self._websocket_helper and self._websocket_helper.IsClosed() and self._websocket_helper.ErrorMsg():
        extra_msg = ''
        if self._websocket_helper.ErrorMsg().startswith('Handshake status 40'):
            extra_msg = ' (May be due to missing permissions)'
        elif self._websocket_helper.ErrorMsg().startswith('4003'):
            extra_msg = ' (Failed to connect to port %d)' % self._tunnel_target.port
        error_msg = 'Error while connecting [%s].%s' % (self._websocket_helper.ErrorMsg(), extra_msg)
        raise ConnectionCreationError(error_msg)
    raise ConnectionCreationError('Unexpected error while connecting. Check logs for more details.')