import contextlib
import os
import time
from threading import Thread
import zmq
from zmq.tests import PYPY, BaseZMQTestCase, SkipTest
from zmq.utils import z85
def zap_handler(self):
    socket = self.context.socket(zmq.REP)
    socket.bind('inproc://zeromq.zap.01')
    try:
        msg = self.recv_multipart(socket)
        version, sequence, domain, address, identity, mechanism = msg[:6]
        if mechanism == b'PLAIN':
            username, password = msg[6:]
        elif mechanism == b'CURVE':
            msg[6]
        assert version == b'1.0'
        assert identity == b'IDENT'
        reply = [version, sequence]
        if mechanism == b'CURVE' or (mechanism == b'PLAIN' and username == USER and (password == PASS)) or mechanism == b'NULL':
            reply.extend([b'200', b'OK', b'anonymous', b'\x05Hello\x00\x00\x00\x05World'])
        else:
            reply.extend([b'400', b'Invalid username or password', b'', b''])
        socket.send_multipart(reply)
    finally:
        socket.close()