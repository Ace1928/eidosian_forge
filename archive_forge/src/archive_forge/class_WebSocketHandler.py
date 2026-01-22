import sys
import struct
import ssl
from base64 import b64encode
from hashlib import sha1
import logging
from socket import error as SocketError
import errno
import threading
from socketserver import ThreadingMixIn, TCPServer, StreamRequestHandler
from websocket_server.thread import WebsocketServerThread
class WebSocketHandler(StreamRequestHandler):

    def __init__(self, socket, addr, server):
        self.server = server
        assert not hasattr(self, '_send_lock'), '_send_lock already exists'
        self._send_lock = threading.Lock()
        if server.key and server.cert:
            try:
                socket = ssl.wrap_socket(socket, server_side=True, certfile=server.cert, keyfile=server.key)
            except:
                logger.warning('SSL not available (are the paths {} and {} correct for the key and cert?)'.format(server.key, server.cert))
        StreamRequestHandler.__init__(self, socket, addr, server)

    def setup(self):
        StreamRequestHandler.setup(self)
        self.keep_alive = True
        self.handshake_done = False
        self.valid_client = False

    def handle(self):
        while self.keep_alive:
            if not self.handshake_done:
                self.handshake()
            elif self.valid_client:
                self.read_next_message()

    def read_bytes(self, num):
        return self.rfile.read(num)

    def read_next_message(self):
        try:
            b1, b2 = self.read_bytes(2)
        except SocketError as e:
            if e.errno == errno.ECONNRESET:
                logger.info('Client closed connection.')
                self.keep_alive = 0
                return
            b1, b2 = (0, 0)
        except ValueError as e:
            b1, b2 = (0, 0)
        fin = b1 & FIN
        opcode = b1 & OPCODE
        masked = b2 & MASKED
        payload_length = b2 & PAYLOAD_LEN
        if opcode == OPCODE_CLOSE_CONN:
            logger.info('Client asked to close connection.')
            self.keep_alive = 0
            return
        if not masked:
            logger.warning('Client must always be masked.')
            self.keep_alive = 0
            return
        if opcode == OPCODE_CONTINUATION:
            logger.warning('Continuation frames are not supported.')
            return
        elif opcode == OPCODE_BINARY:
            logger.warning('Binary frames are not supported.')
            return
        elif opcode == OPCODE_TEXT:
            opcode_handler = self.server._message_received_
        elif opcode == OPCODE_PING:
            opcode_handler = self.server._ping_received_
        elif opcode == OPCODE_PONG:
            opcode_handler = self.server._pong_received_
        else:
            logger.warning('Unknown opcode %#x.' % opcode)
            self.keep_alive = 0
            return
        if payload_length == 126:
            payload_length = struct.unpack('>H', self.rfile.read(2))[0]
        elif payload_length == 127:
            payload_length = struct.unpack('>Q', self.rfile.read(8))[0]
        masks = self.read_bytes(4)
        message_bytes = bytearray()
        for message_byte in self.read_bytes(payload_length):
            message_byte ^= masks[len(message_bytes) % 4]
            message_bytes.append(message_byte)
        opcode_handler(self, message_bytes.decode('utf8'))

    def send_message(self, message):
        self.send_text(message)

    def send_pong(self, message):
        self.send_text(message, OPCODE_PONG)

    def send_close(self, status=CLOSE_STATUS_NORMAL, reason=DEFAULT_CLOSE_REASON):
        """
        Send CLOSE to client

        Args:
            status: Status as defined in https://datatracker.ietf.org/doc/html/rfc6455#section-7.4.1
            reason: Text with reason of closing the connection
        """
        if status < CLOSE_STATUS_NORMAL or status > 1015:
            raise Exception(f'CLOSE status must be between 1000 and 1015, got {status}')
        header = bytearray()
        payload = struct.pack('!H', status) + reason
        payload_length = len(payload)
        assert payload_length <= 125, 'We only support short closing reasons at the moment'
        header.append(FIN | OPCODE_CLOSE_CONN)
        header.append(payload_length)
        with self._send_lock:
            self.request.send(header + payload)

    def send_text(self, message, opcode=OPCODE_TEXT):
        """
        Important: Fragmented(=continuation) messages are not supported since
        their usage cases are limited - when we don't know the payload length.
        """
        if isinstance(message, bytes):
            message = try_decode_UTF8(message)
            if not message:
                logger.warning("Can't send message, message is not valid UTF-8")
                return False
        elif not isinstance(message, str):
            logger.warning("Can't send message, message has to be a string or bytes. Got %s" % type(message))
            return False
        header = bytearray()
        payload = encode_to_UTF8(message)
        payload_length = len(payload)
        if payload_length <= 125:
            header.append(FIN | opcode)
            header.append(payload_length)
        elif payload_length >= 126 and payload_length <= 65535:
            header.append(FIN | opcode)
            header.append(PAYLOAD_LEN_EXT16)
            header.extend(struct.pack('>H', payload_length))
        elif payload_length < 18446744073709551616:
            header.append(FIN | opcode)
            header.append(PAYLOAD_LEN_EXT64)
            header.extend(struct.pack('>Q', payload_length))
        else:
            raise Exception('Message is too big. Consider breaking it into chunks.')
            return
        with self._send_lock:
            self.request.send(header + payload)

    def read_http_headers(self):
        headers = {}
        http_get = self.rfile.readline().decode().strip()
        assert http_get.upper().startswith('GET')
        while True:
            header = self.rfile.readline().decode().strip()
            if not header:
                break
            head, value = header.split(':', 1)
            headers[head.lower().strip()] = value.strip()
        return headers

    def handshake(self):
        headers = self.read_http_headers()
        try:
            assert headers['upgrade'].lower() == 'websocket'
        except AssertionError:
            self.keep_alive = False
            return
        try:
            key = headers['sec-websocket-key']
        except KeyError:
            logger.warning('Client tried to connect but was missing a key')
            self.keep_alive = False
            return
        response = self.make_handshake_response(key)
        with self._send_lock:
            self.handshake_done = self.request.send(response.encode())
        self.valid_client = True
        self.server._new_client_(self)

    @classmethod
    def make_handshake_response(cls, key):
        return 'HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: %s\r\n\r\n' % cls.calculate_response_key(key)

    @classmethod
    def calculate_response_key(cls, key):
        GUID = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11'
        hash = sha1(key.encode() + GUID.encode())
        response_key = b64encode(hash.digest()).strip()
        return response_key.decode('ASCII')

    def finish(self):
        self.server._client_left_(self)