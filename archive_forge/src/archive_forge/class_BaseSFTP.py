import select
import socket
import struct
from paramiko import util
from paramiko.common import DEBUG, byte_chr, byte_ord
from paramiko.message import Message
class BaseSFTP:

    def __init__(self):
        self.logger = util.get_logger('paramiko.sftp')
        self.sock = None
        self.ultra_debug = False

    def _send_version(self):
        m = Message()
        m.add_int(_VERSION)
        self._send_packet(CMD_INIT, m)
        t, data = self._read_packet()
        if t != CMD_VERSION:
            raise SFTPError('Incompatible sftp protocol')
        version = struct.unpack('>I', data[:4])[0]
        return version

    def _send_server_version(self):
        t, data = self._read_packet()
        if t != CMD_INIT:
            raise SFTPError('Incompatible sftp protocol')
        version = struct.unpack('>I', data[:4])[0]
        extension_pairs = ['check-file', 'md5,sha1']
        msg = Message()
        msg.add_int(_VERSION)
        msg.add(*extension_pairs)
        self._send_packet(CMD_VERSION, msg)
        return version

    def _log(self, level, msg, *args):
        self.logger.log(level, msg, *args)

    def _write_all(self, out):
        while len(out) > 0:
            n = self.sock.send(out)
            if n <= 0:
                raise EOFError()
            if n == len(out):
                return
            out = out[n:]
        return

    def _read_all(self, n):
        out = bytes()
        while n > 0:
            if isinstance(self.sock, socket.socket):
                while True:
                    read, write, err = select.select([self.sock], [], [], 0.1)
                    if len(read) > 0:
                        x = self.sock.recv(n)
                        break
            else:
                x = self.sock.recv(n)
            if len(x) == 0:
                raise EOFError()
            out += x
            n -= len(x)
        return out

    def _send_packet(self, t, packet):
        packet = packet.asbytes()
        out = struct.pack('>I', len(packet) + 1) + byte_chr(t) + packet
        if self.ultra_debug:
            self._log(DEBUG, util.format_binary(out, 'OUT: '))
        self._write_all(out)

    def _read_packet(self):
        x = self._read_all(4)
        if byte_ord(x[0]):
            raise SFTPError('Garbage packet received')
        size = struct.unpack('>I', x)[0]
        data = self._read_all(size)
        if self.ultra_debug:
            self._log(DEBUG, util.format_binary(data, 'IN: '))
        if size > 0:
            t = byte_ord(data[0])
            return (t, data[1:])
        return (0, bytes())