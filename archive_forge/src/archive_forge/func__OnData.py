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
def _OnData(self, binary_data):
    """Receive a single message from the server."""
    tag, bytes_left = utils.ExtractSubprotocolTag(binary_data)
    if tag == utils.SUBPROTOCOL_TAG_DATA:
        self._HandleSubprotocolData(bytes_left)
    elif tag == utils.SUBPROTOCOL_TAG_ACK:
        self._HandleSubprotocolAck(bytes_left)
    elif tag == utils.SUBPROTOCOL_TAG_CONNECT_SUCCESS_SID:
        self._HandleSubprotocolConnectSuccessSid(bytes_left)
    elif tag == utils.SUBPROTOCOL_TAG_RECONNECT_SUCCESS_ACK:
        self._HandleSubprotocolReconnectSuccessAck(bytes_left)
    else:
        log.debug('Unsupported subprotocol tag [%r], discarding the message', tag)