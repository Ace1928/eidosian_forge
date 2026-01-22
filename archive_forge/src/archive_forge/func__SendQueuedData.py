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
def _SendQueuedData(self):
    """Send data that is sitting in the unsent data queue."""
    try:
        while not self._stopping:
            self._MaybeSendAck()
            try:
                if not self._data_to_resend.empty():
                    data = self._data_to_resend.get()
                else:
                    data = self._unsent_data.get(timeout=MAX_WEBSOCKET_SEND_WAIT_TIME_SEC)
            except queue.Empty:
                if self._IsClosed():
                    raise helper.WebSocketConnectionClosed
                break
            if data is EOFError or data is StoppingError:
                self._stopping = True
                if data is EOFError:
                    self._input_eof = True
                break
            if data is SendAckNotification:
                self._SendAck()
                continue
            self._unconfirmed_data.append(data)
            send_data = utils.CreateSubprotocolDataFrame(data)
            self._websocket_helper.Send(send_data)
    finally:
        if self._input_eof and self._data_to_resend.empty() and self._unsent_data.empty():
            self._sent_all.set()