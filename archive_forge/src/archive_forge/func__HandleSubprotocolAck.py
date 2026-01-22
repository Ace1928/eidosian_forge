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
def _HandleSubprotocolAck(self, binary_data):
    """Handle Subprotocol ACK Frame."""
    if not self._HasConnected():
        self._StopConnectionAsync()
        raise SubprotocolEarlyAckError('Received ACK before connected.')
    bytes_confirmed, bytes_left = utils.ExtractSubprotocolAck(binary_data)
    self._ConfirmData(bytes_confirmed)
    if bytes_left:
        log.debug('[%d] Discarding [%d] extra bytes after processing ACK', self._conn_id, len(bytes_left))