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
def WaitForAllSent(self):
    """Wait until all local data has been sent on the websocket.

    Blocks until either all data from Send() has been sent, or it times out
    waiting. Once true, always returns true. Even if this returns true, a
    reconnect could occur causing previously sent data to be resent. Must only
    be called after an EOF has been given to Send().

    Returns:
      True on success, False on timeout.
    """
    return self._sent_all.wait(ALL_DATA_SENT_WAIT_TIME_SEC)