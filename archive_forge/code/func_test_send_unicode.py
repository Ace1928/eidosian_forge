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
def test_send_unicode(self):
    """test sending unicode objects"""
    a, b = self.create_bound_pair(zmq.PAIR, zmq.PAIR)
    self.sockets.extend([a, b])
    u = 'çπ§'
    self.assertRaises(TypeError, a.send, u, copy=False)
    self.assertRaises(TypeError, a.send, u, copy=True)
    a.send_unicode(u)
    s = b.recv()
    assert s == u.encode('utf8')
    assert s.decode('utf8') == u
    a.send_unicode(u, encoding='utf16')
    s = b.recv_unicode(encoding='utf16')
    assert s == u