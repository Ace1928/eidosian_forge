import base64
import codecs
import collections
import errno
from random import Random
from socket import error as SocketError
import string
import struct
import sys
import time
import zlib
from eventlet import semaphore
from eventlet import wsgi
from eventlet.green import socket
from eventlet.support import get_errno
class RFC6455WebSocket(WebSocket):

    def __init__(self, sock, environ, version=13, protocol=None, client=False, extensions=None, max_frame_length=DEFAULT_MAX_FRAME_LENGTH):
        super().__init__(sock, environ, version)
        self.iterator = self._iter_frames()
        self.client = client
        self.protocol = protocol
        self.extensions = extensions or {}
        self._deflate_enc = None
        self._deflate_dec = None
        self.max_frame_length = max_frame_length
        self._remote_close_data = None

    class UTF8Decoder:

        def __init__(self):
            if utf8validator:
                self.validator = utf8validator.Utf8Validator()
            else:
                self.validator = None
            decoderclass = codecs.getincrementaldecoder('utf8')
            self.decoder = decoderclass()

        def reset(self):
            if self.validator:
                self.validator.reset()
            self.decoder.reset()

        def decode(self, data, final=False):
            if self.validator:
                valid, eocp, c_i, t_i = self.validator.validate(data)
                if not valid:
                    raise ValueError('Data is not valid unicode')
            return self.decoder.decode(data, final)

    def _get_permessage_deflate_enc(self):
        options = self.extensions.get('permessage-deflate')
        if options is None:
            return None

        def _make():
            return zlib.compressobj(zlib.Z_DEFAULT_COMPRESSION, zlib.DEFLATED, -options.get('client_max_window_bits' if self.client else 'server_max_window_bits', zlib.MAX_WBITS))
        if options.get('client_no_context_takeover' if self.client else 'server_no_context_takeover'):
            return _make()
        else:
            if self._deflate_enc is None:
                self._deflate_enc = _make()
            return self._deflate_enc

    def _get_permessage_deflate_dec(self, rsv1):
        options = self.extensions.get('permessage-deflate')
        if options is None or not rsv1:
            return None

        def _make():
            return zlib.decompressobj(-options.get('server_max_window_bits' if self.client else 'client_max_window_bits', zlib.MAX_WBITS))
        if options.get('server_no_context_takeover' if self.client else 'client_no_context_takeover'):
            return _make()
        else:
            if self._deflate_dec is None:
                self._deflate_dec = _make()
            return self._deflate_dec

    def _get_bytes(self, numbytes):
        data = b''
        while len(data) < numbytes:
            d = self.socket.recv(numbytes - len(data))
            if not d:
                raise ConnectionClosedError()
            data = data + d
        return data

    class Message:

        def __init__(self, opcode, max_frame_length, decoder=None, decompressor=None):
            self.decoder = decoder
            self.data = []
            self.finished = False
            self.opcode = opcode
            self.decompressor = decompressor
            self.max_frame_length = max_frame_length

        def push(self, data, final=False):
            self.finished = final
            self.data.append(data)

        def getvalue(self):
            data = b''.join(self.data)
            if not self.opcode & 8 and self.decompressor:
                data = self.decompressor.decompress(data + b'\x00\x00\xff\xff', self.max_frame_length)
                if self.decompressor.unconsumed_tail:
                    raise FailedConnectionError(1009, 'Incoming compressed frame exceeds length limit of {} bytes.'.format(self.max_frame_length))
            if self.decoder:
                data = self.decoder.decode(data, self.finished)
            return data

    @staticmethod
    def _apply_mask(data, mask, length=None, offset=0):
        if length is None:
            length = len(data)
        cnt = range(length)
        return b''.join((bytes((data[i] ^ mask[(offset + i) % 4],)) for i in cnt))

    def _handle_control_frame(self, opcode, data):
        if opcode == 8:
            self._remote_close_data = data
            if not data:
                status = 1000
            elif len(data) > 1:
                status = struct.unpack_from('!H', data)[0]
                if not status or status not in VALID_CLOSE_STATUS:
                    raise FailedConnectionError(1002, 'Unexpected close status code.')
                try:
                    data = self.UTF8Decoder().decode(data[2:], True)
                except (UnicodeDecodeError, ValueError):
                    raise FailedConnectionError(1002, 'Close message data should be valid UTF-8.')
            else:
                status = 1002
            self.close(close_data=(status, ''))
            raise ConnectionClosedError()
        elif opcode == 9:
            self.send(data, control_code=10)
        elif opcode == 10:
            pass
        else:
            raise FailedConnectionError(1002, 'Unknown control frame received.')

    def _iter_frames(self):
        fragmented_message = None
        try:
            while True:
                message = self._recv_frame(message=fragmented_message)
                if message.opcode & 8:
                    self._handle_control_frame(message.opcode, message.getvalue())
                    continue
                if fragmented_message and message is not fragmented_message:
                    raise RuntimeError('Unexpected message change.')
                fragmented_message = message
                if message.finished:
                    data = fragmented_message.getvalue()
                    fragmented_message = None
                    yield data
        except FailedConnectionError:
            exc_typ, exc_val, exc_tb = sys.exc_info()
            self.close(close_data=(exc_val.status, exc_val.message))
        except ConnectionClosedError:
            return
        except Exception:
            self.close(close_data=(1011, 'Internal Server Error'))
            raise

    def _recv_frame(self, message=None):
        recv = self._get_bytes
        header = recv(2)
        a, b = struct.unpack('!BB', header)
        finished = a >> 7 == 1
        rsv123 = a >> 4 & 7
        rsv1 = rsv123 & 4
        if rsv123:
            if rsv1 and 'permessage-deflate' not in self.extensions:
                raise FailedConnectionError(1002, 'RSV1, RSV2, RSV3: MUST be 0 unless an extension is negotiated that defines meanings for non-zero values.')
        opcode = a & 15
        if opcode not in (0, 1, 2, 8, 9, 10):
            raise FailedConnectionError(1002, 'Unknown opcode received.')
        masked = b & 128 == 128
        if not masked and (not self.client):
            raise FailedConnectionError(1002, 'A client MUST mask all frames that it sends to the server')
        length = b & 127
        if opcode & 8:
            if not finished:
                raise FailedConnectionError(1002, 'Control frames must not be fragmented.')
            if length > 125:
                raise FailedConnectionError(1002, 'All control frames MUST have a payload length of 125 bytes or less')
        elif opcode and message:
            raise FailedConnectionError(1002, 'Received a non-continuation opcode within fragmented message.')
        elif not opcode and (not message):
            raise FailedConnectionError(1002, 'Received continuation opcode with no previous fragments received.')
        if length == 126:
            length = struct.unpack('!H', recv(2))[0]
        elif length == 127:
            length = struct.unpack('!Q', recv(8))[0]
        if length > self.max_frame_length:
            raise FailedConnectionError(1009, 'Incoming frame of {} bytes is above length limit of {} bytes.'.format(length, self.max_frame_length))
        if masked:
            mask = struct.unpack('!BBBB', recv(4))
        received = 0
        if not message or opcode & 8:
            decoder = self.UTF8Decoder() if opcode == 1 else None
            decompressor = self._get_permessage_deflate_dec(rsv1)
            message = self.Message(opcode, self.max_frame_length, decoder=decoder, decompressor=decompressor)
        if not length:
            message.push(b'', final=finished)
        else:
            while received < length:
                d = self.socket.recv(length - received)
                if not d:
                    raise ConnectionClosedError()
                dlen = len(d)
                if masked:
                    d = self._apply_mask(d, mask, length=dlen, offset=received)
                received = received + dlen
                try:
                    message.push(d, final=finished)
                except (UnicodeDecodeError, ValueError):
                    raise FailedConnectionError(1007, 'Text data must be valid utf-8')
        return message

    def _pack_message(self, message, masked=False, continuation=False, final=True, control_code=None):
        is_text = False
        if isinstance(message, str):
            message = message.encode('utf-8')
            is_text = True
        compress_bit = 0
        compressor = self._get_permessage_deflate_enc()
        is_control_frame = (control_code or 0) & 8
        if message and compressor and (not is_control_frame):
            message = compressor.compress(message)
            message += compressor.flush(zlib.Z_SYNC_FLUSH)
            assert message[-4:] == b'\x00\x00\xff\xff'
            message = message[:-4]
            compress_bit = 1 << 6
        length = len(message)
        if not length:
            masked = False
        if control_code:
            if control_code not in (8, 9, 10):
                raise ProtocolError('Unknown control opcode.')
            if continuation or not final:
                raise ProtocolError('Control frame cannot be a fragment.')
            if length > 125:
                raise ProtocolError('Control frame data too large (>125).')
            header = struct.pack('!B', control_code | 1 << 7)
        else:
            opcode = 0 if continuation else (1 if is_text else 2) | compress_bit
            header = struct.pack('!B', opcode | (1 << 7 if final else 0))
        lengthdata = 1 << 7 if masked else 0
        if length > 65535:
            lengthdata = struct.pack('!BQ', lengthdata | 127, length)
        elif length > 125:
            lengthdata = struct.pack('!BH', lengthdata | 126, length)
        else:
            lengthdata = struct.pack('!B', lengthdata | length)
        if masked:
            rand = Random(time.time())
            mask = [rand.getrandbits(8) for _ in range(4)]
            message = RFC6455WebSocket._apply_mask(message, mask, length)
            maskdata = struct.pack('!BBBB', *mask)
        else:
            maskdata = b''
        return b''.join((header, lengthdata, maskdata, message))

    def wait(self):
        for i in self.iterator:
            return i

    def _send(self, frame):
        self._sendlock.acquire()
        try:
            self.socket.sendall(frame)
        finally:
            self._sendlock.release()

    def send(self, message, **kw):
        kw['masked'] = self.client
        payload = self._pack_message(message, **kw)
        self._send(payload)

    def _send_closing_frame(self, ignore_send_errors=False, close_data=None):
        if self.version in (8, 13) and (not self.websocket_closed):
            if close_data is not None:
                status, msg = close_data
                if isinstance(msg, str):
                    msg = msg.encode('utf-8')
                data = struct.pack('!H', status) + msg
            else:
                data = ''
            try:
                self.send(data, control_code=8)
            except OSError:
                if not ignore_send_errors:
                    raise
            self.websocket_closed = True

    def close(self, close_data=None):
        """Forcibly close the websocket; generally it is preferable to
        return from the handler method."""
        try:
            self._send_closing_frame(close_data=close_data, ignore_send_errors=True)
            self.socket.shutdown(socket.SHUT_WR)
        except OSError as e:
            if e.errno != errno.ENOTCONN:
                self.log.write('{ctx} socket shutdown error: {e}'.format(ctx=self.log_context, e=e))
        finally:
            self.socket.close()