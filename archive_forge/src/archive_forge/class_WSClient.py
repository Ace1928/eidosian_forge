from kubernetes.client.rest import ApiException
import select
import certifi
import time
import collections
from websocket import WebSocket, ABNF, enableTrace
import six
import ssl
from six.moves.urllib.parse import urlencode, quote_plus, urlparse, urlunparse
class WSClient:

    def __init__(self, configuration, url, headers):
        """A websocket client with support for channels.

            Exec command uses different channels for different streams. for
        example, 0 is stdin, 1 is stdout and 2 is stderr. Some other API calls
        like port forwarding can forward different pods' streams to different
        channels.
        """
        enableTrace(False)
        header = []
        self._connected = False
        self._channels = {}
        self._all = ''
        if headers and 'authorization' in headers:
            header.append('authorization: %s' % headers['authorization'])
        if headers and 'sec-websocket-protocol' in headers:
            header.append('sec-websocket-protocol: %s' % headers['sec-websocket-protocol'])
        else:
            header.append('sec-websocket-protocol: v4.channel.k8s.io')
        if url.startswith('wss://') and configuration.verify_ssl:
            ssl_opts = {'cert_reqs': ssl.CERT_REQUIRED, 'ca_certs': configuration.ssl_ca_cert or certifi.where()}
            if configuration.assert_hostname is not None:
                ssl_opts['check_hostname'] = configuration.assert_hostname
        else:
            ssl_opts = {'cert_reqs': ssl.CERT_NONE}
        if configuration.cert_file:
            ssl_opts['certfile'] = configuration.cert_file
        if configuration.key_file:
            ssl_opts['keyfile'] = configuration.key_file
        self.sock = WebSocket(sslopt=ssl_opts, skip_utf8_validation=False)
        self.sock.connect(url, header=header)
        self._connected = True

    def peek_channel(self, channel, timeout=0):
        """Peek a channel and return part of the input,

        empty string otherwise.
    """
        self.update(timeout=timeout)
        if channel in self._channels:
            return self._channels[channel]
        return ''

    def read_channel(self, channel, timeout=0):
        """Read data from a channel."""
        if channel not in self._channels:
            ret = self.peek_channel(channel, timeout)
        else:
            ret = self._channels[channel]
        if channel in self._channels:
            del self._channels[channel]
        return ret

    def readline_channel(self, channel, timeout=None):
        """Read a line from a channel."""
        if timeout is None:
            timeout = float('inf')
        start = time.time()
        while self.is_open() and time.time() - start < timeout:
            if channel in self._channels:
                data = self._channels[channel]
                if '\n' in data:
                    index = data.find('\n')
                    ret = data[:index]
                    data = data[index + 1:]
                    if data:
                        self._channels[channel] = data
                    else:
                        del self._channels[channel]
                    return ret
            self.update(timeout=timeout - time.time() + start)

    def write_channel(self, channel, data):
        """Write data to a channel."""
        self.sock.send(chr(channel) + data)

    def peek_stdout(self, timeout=0):
        """Same as peek_channel with channel=1."""
        return self.peek_channel(STDOUT_CHANNEL, timeout=timeout)

    def read_stdout(self, timeout=None):
        """Same as read_channel with channel=1."""
        return self.read_channel(STDOUT_CHANNEL, timeout=timeout)

    def readline_stdout(self, timeout=None):
        """Same as readline_channel with channel=1."""
        return self.readline_channel(STDOUT_CHANNEL, timeout=timeout)

    def peek_stderr(self, timeout=0):
        """Same as peek_channel with channel=2."""
        return self.peek_channel(STDERR_CHANNEL, timeout=timeout)

    def read_stderr(self, timeout=None):
        """Same as read_channel with channel=2."""
        return self.read_channel(STDERR_CHANNEL, timeout=timeout)

    def readline_stderr(self, timeout=None):
        """Same as readline_channel with channel=2."""
        return self.readline_channel(STDERR_CHANNEL, timeout=timeout)

    def read_all(self):
        """Return buffered data received on stdout and stderr channels.

        This is useful for non-interactive call where a set of command passed
        to the API call and their result is needed after the call is concluded.
        Should be called after run_forever() or update()

        TODO: Maybe we can process this and return a more meaningful map with
        channels mapped for each input.
        """
        out = self._all
        self._all = ''
        self._channels = {}
        return out

    def is_open(self):
        """True if the connection is still alive."""
        return self._connected

    def write_stdin(self, data):
        """The same as write_channel with channel=0."""
        self.write_channel(STDIN_CHANNEL, data)

    def update(self, timeout=0):
        """Update channel buffers with at most one complete frame of input."""
        if not self.is_open():
            return
        if not self.sock.connected:
            self._connected = False
            return
        r, _, _ = select.select((self.sock.sock,), (), (), timeout)
        if r:
            op_code, frame = self.sock.recv_data_frame(True)
            if op_code == ABNF.OPCODE_CLOSE:
                self._connected = False
                return
            elif op_code == ABNF.OPCODE_BINARY or op_code == ABNF.OPCODE_TEXT:
                data = frame.data
                if six.PY3:
                    data = data.decode('utf-8', 'replace')
                if len(data) > 1:
                    channel = ord(data[0])
                    data = data[1:]
                    if data:
                        if channel in [STDOUT_CHANNEL, STDERR_CHANNEL]:
                            self._all += data
                        if channel not in self._channels:
                            self._channels[channel] = data
                        else:
                            self._channels[channel] += data

    def run_forever(self, timeout=None):
        """Wait till connection is closed or timeout reached.

    Buffer any input
        received during this time.
    """
        if timeout:
            start = time.time()
            while self.is_open() and time.time() - start < timeout:
                self.update(timeout=timeout - time.time() + start)
        else:
            while self.is_open():
                self.update(timeout=None)

    def close(self, **kwargs):
        """
        close websocket connection.
        """
        self._connected = False
        if self.sock:
            self.sock.close(**kwargs)