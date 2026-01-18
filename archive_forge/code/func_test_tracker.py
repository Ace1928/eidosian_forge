import copy
import errno
import json
import os
import platform
import socket
import sys
import time
import warnings
from unittest import mock
import pytest
from pytest import mark
import zmq
from zmq.tests import BaseZMQTestCase, GreenTest, SkipTest, have_gevent, skip_pypy
@skip_pypy
def test_tracker(self):
    """test the MessageTracker object for tracking when zmq is done with a buffer"""
    addr = 'tcp://127.0.0.1'
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    iface = '%s:%i' % (addr, port)
    sock.close()
    time.sleep(0.1)
    a = self.context.socket(zmq.PUSH)
    b = self.context.socket(zmq.PULL)
    self.sockets.extend([a, b])
    a.connect(iface)
    time.sleep(0.1)
    p1 = a.send(b'something', copy=False, track=True)
    assert isinstance(p1, zmq.MessageTracker)
    assert p1 is zmq._FINISHED_TRACKER
    assert p1.done
    a.copy_threshold = 0
    p2 = a.send_multipart([b'something', b'else'], copy=False, track=True)
    assert isinstance(p2, zmq.MessageTracker)
    assert not p2.done
    b.bind(iface)
    msg = self.recv_multipart(b)
    for i in range(10):
        if p1.done:
            break
        time.sleep(0.1)
    assert p1.done is True
    assert msg == [b'something']
    msg = self.recv_multipart(b)
    for i in range(10):
        if p2.done:
            break
        time.sleep(0.1)
    assert p2.done is True
    assert msg == [b'something', b'else']
    m = zmq.Frame(b'again', copy=False, track=True)
    assert m.tracker.done is False
    p1 = a.send(m, copy=False)
    p2 = a.send(m, copy=False)
    assert m.tracker.done is False
    assert p1.done is False
    assert p2.done is False
    msg = self.recv_multipart(b)
    assert m.tracker.done is False
    assert msg == [b'again']
    msg = self.recv_multipart(b)
    assert m.tracker.done is False
    assert msg == [b'again']
    assert p1.done is False
    assert p2.done is False
    m.tracker
    del m
    for i in range(10):
        if p1.done:
            break
        time.sleep(0.1)
    assert p1.done is True
    assert p2.done is True
    m = zmq.Frame(b'something', track=False)
    self.assertRaises(ValueError, a.send, m, copy=False, track=True)