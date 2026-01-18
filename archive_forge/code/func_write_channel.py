from kubernetes.client.rest import ApiException
import select
import certifi
import time
import collections
from websocket import WebSocket, ABNF, enableTrace
import six
import ssl
from six.moves.urllib.parse import urlencode, quote_plus, urlparse, urlunparse
def write_channel(self, channel, data):
    """Write data to a channel."""
    self.sock.send(chr(channel) + data)