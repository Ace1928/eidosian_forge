from kubernetes.client.rest import ApiException
import select
import certifi
import time
import collections
from websocket import WebSocket, ABNF, enableTrace
import six
import ssl
from six.moves.urllib.parse import urlencode, quote_plus, urlparse, urlunparse
def write_stdin(self, data):
    """The same as write_channel with channel=0."""
    self.write_channel(STDIN_CHANNEL, data)