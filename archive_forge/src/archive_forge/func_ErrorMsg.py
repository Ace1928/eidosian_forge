from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import logging
import ssl
import sys
import threading
import traceback
from googlecloudsdk.api_lib.compute import iap_tunnel_lightweight_websocket as iap_websocket
from googlecloudsdk.api_lib.compute import iap_tunnel_websocket_utils as utils
from googlecloudsdk.core import context_aware
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.util import encoding
import six
import websocket
def ErrorMsg(self):
    return self._error_msg