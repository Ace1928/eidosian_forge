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
def _MaybeSendAck(self):
    """Decide if an ACK should be sent back to the server."""
    if self._cant_send_ack.is_set():
        return
    total_bytes = self._total_bytes_received
    bytes_recv_and_ackd = self._total_bytes_received_and_acked
    window_size = utils.SUBPROTOCOL_MAX_DATA_FRAME_SIZE
    if total_bytes - bytes_recv_and_ackd > 2 * window_size:
        self._cant_send_ack.set()
        self._unsent_data.put(SendAckNotification)