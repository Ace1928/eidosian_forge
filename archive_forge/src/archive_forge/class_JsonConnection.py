import base64
import errno
import json
from multiprocessing import connection
from multiprocessing import managers
import socket
import struct
import weakref
from oslo_rootwrap import wrapper
class JsonConnection(object):

    def __init__(self, sock):
        sock.setblocking(True)
        self._socket = sock

    def send_bytes(self, s):
        self._socket.sendall(struct.pack('!Q', len(s)))
        self._socket.sendall(s)

    def recv_bytes(self, maxsize=None):
        item = struct.unpack('!Q', self.recvall(8))[0]
        if maxsize is not None and item > maxsize:
            raise RuntimeError('Too big message received')
        s = self.recvall(item)
        return s

    def send(self, obj):
        s = self.dumps(obj)
        self.send_bytes(s)

    def recv(self):
        s = self.recv_bytes()
        return self.loads(s)

    def close(self):
        self._socket.close()

    def half_close(self):
        self._socket.shutdown(socket.SHUT_RD)

    def _recvall_slow(self, size):
        remaining = size
        res = []
        while remaining:
            piece = self._socket.recv(remaining)
            if not piece:
                raise EOFError
            res.append(piece)
            remaining -= len(piece)
        return b''.join(res)

    def recvall(self, size):
        buf = bytearray(size)
        mem = memoryview(buf)
        got = 0
        while got < size:
            piece_size = self._socket.recv_into(mem[got:])
            if not piece_size:
                raise EOFError
            got += piece_size
        return bytes(buf)

    @staticmethod
    def dumps(obj):
        return json.dumps(obj, cls=RpcJSONEncoder).encode('utf-8')

    @staticmethod
    def loads(s):
        res = json.loads(s.decode('utf-8'), object_hook=rpc_object_hook)
        try:
            kind = res[0]
        except (IndexError, TypeError):
            pass
        else:
            if kind in ('#TRACEBACK', '#UNSERIALIZABLE') and (not isinstance(res[1], str)):
                res[1] = res[1].encode('utf-8', 'replace')
        return res